from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextContainer
from PIL import Image
import cv2
# from pdf2image import convert_from_path
import os
import sys
from tqdm import tqdm
# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)
from PIL import Image, ImageDraw
# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import glob
import ast

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, Trainer_ARSL
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.utils.logger import setup_logger

logger = setup_logger('train')


# All logic starting from here

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    args = parser.parse_args()
    return args

def get_png_file_paths(directory):
    png_paths = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".png"):
                png_paths.append(os.path.join(directory, filename))
    except Exception as e:
        print(f"Error reading PNG file paths from directory: {e}")
    return png_paths

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def scale_bbox_coordinates(bbox, image_width, image_height):
    # Bounding box coordinates are [x_min, y_min, width, height]
    scaled_bbox = [
        bbox[0] / image_width,   # x_min
        bbox[1] / image_height,  # y_min
        bbox[2] / image_width,           
        bbox[3] / image_height           
    ]

    return scaled_bbox



def extract_data_from_png_folder(cfg, path):
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)

    png_list = get_png_file_paths(path)
    
    # Initialize a dictionary to store image dimensions
    image_dimensions = {}
    
    # Get dimensions for each image
    for image_path in png_list:
        image_width, image_height = get_image_dimensions(image_path)
        image_dimensions[image_path] = (image_width, image_height)

    # print('IMage dimension', image_dimensions)
    layout_structure = trainer.get_bboxes(png_list)
    
    # Scale bounding box coordinates for each image
    for image_result in layout_structure:
        for item in image_result:
            bbox = item['bbox']
            image_path = png_list[item['image_id']]
            image_width, image_height = image_dimensions[image_path]
            item['bbox_scale'] = scale_bbox_coordinates(bbox, image_width, image_height)
            item['image_filename'] = os.path.basename(image_path)
        # Sort bounding box coordinates based on the y-coordinate of the top-left corner
        image_result.sort(key=lambda x: x['bbox_scale'][1])
    # print(layout_structure)
    return layout_structure


def extract_data_from_pdf_folder(folder_path):
    pdf_data = []
    idx = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_file = os.path.join(folder_path, filename)            
            for page_layout in extract_pages(pdf_file):
                _, _, width, height = page_layout.bbox  # Extract page width and height
                # print(width, height)
                image_name = os.path.splitext(filename)[0]
                page_data = []
                for element in page_layout:
                    if isinstance(element, LTTextBox):
                        bbox = element.bbox
                        text = element.get_text().strip()
                        # print('')
                        # Calculate scaled bounding box values and (0,0) stay on top left
                        x_left, y_bot, x_right, y_top = bbox[0], bbox[1], bbox[2], bbox[3]
                        
                        # print(x_left, y_bot, x_right, y_top)
                        y_bot = height - y_bot
                        y_top = height - y_top

                        # print(x_left, y_bot, x_right, y_top)

                        
                        x_left_per = x_left / width
                        x_right_per = x_right / width
                        y_bot_per = y_bot / height
                        y_top_per = y_top / height

                        # print(x_left_per, y_bot_per, x_right_per, y_top_per)
                        # print("------------\n")


                        bbox_scale = (x_left_per, y_top_per, (x_right_per - x_left_per), (y_bot_per - y_top_per))

                        page_data.append({
                            'image_id': idx,
                            'image_name': image_name,
                            'bbox': bbox,
                            'text': text,
                            'bbox_scale': bbox_scale
                        })
                pdf_data.append(page_data)
                idx += 1
    return pdf_data


def reorder_bbox_top_to_bottom(data):
    # Iterate over each page
    reordered_data = []
    for page in data:
        # Sort the bounding boxes based on the y-coordinate of the top-left corner
        sorted_page = sorted(page, key=lambda x: x['bbox'][1])
        reordered_data.append(sorted_page)
    return reordered_data

def is_bbox_inside_image(image_bbox, pdf_bbox):
    width, height = get_image_dimensions('data_preprocessing/pages/page_18.png')
    # print(width, height)
    # print('image Box', image_bbox[0]*width, image_bbox[1]*height, image_bbox[0]*width  + image_bbox[2]*width, image_bbox[1]*height + image_bbox[3]*height)
    # print('PDF Box', pdf_bbox[0]*width, pdf_bbox[1]*height, pdf_bbox[0]*width  + pdf_bbox[2]*width, pdf_bbox[1]*height + pdf_bbox[3]*height)
    return (image_bbox[0]*width - 20  < pdf_bbox[0]*width   and
            image_bbox[1]*height - 20 < pdf_bbox[1]*height  and
            image_bbox[0]*width  + image_bbox[2]*width  + 20  > pdf_bbox[0]*width    + pdf_bbox[2]*width and
            image_bbox[1]*height + image_bbox[3]*height + 20 > pdf_bbox[1]*height   + pdf_bbox[3]*height)

def sort_column1(columns):
    return sorted(columns, key=lambda x: x['bbox_scale'][1])

def sort_column2(columns, division_position ):
    sort_y_cordinate = sorted(columns, key=lambda x: x['bbox_scale'][1])
    left_part = []
    right_part = []
    for item in sort_y_cordinate:
        bbox = item['bbox_scale']
        # Check if the bounding box is on the left or right side of the page
        if bbox[0] < division_position: 
            left_part.append(item)
        else:
            right_part.append(item)
    result = left_part + right_part
    return result

def sort_column3(columns, division_position1, division_position2 ):
    sort_y_cordinate = sorted(columns, key=lambda x: x['bbox_scale'][1])
    left_part = []
    middle_part = []
    right_part = []
    for item in sort_y_cordinate:
        bbox = item['bbox_scale']
        # Check if the bounding box is on the left, middle, or right side of the page
        if bbox[0] < division_position1:
            left_part.append(item)
        elif bbox[0] < division_position2:
            middle_part.append(item)
        else:
            right_part.append(item)
    result = left_part + middle_part + right_part
    return result

def process_data(data_png, data_pdf):
    size_of_data = len(data_png)
    idx = 0
    
    while(idx < size_of_data):
        for images, pdfs in zip(data_png, data_pdf):
            for image in images:
                if image['category_id'] == 1:
                    cols1 = []
                    image_bbox = image['bbox_scale']
                    for pdf in pdfs:
                        pdf_bbox = pdf['bbox_scale']
                        if is_bbox_inside_image(image_bbox, pdf_bbox):
                            cols1.append(pdf)
                        else:
                            pass
                    sort_cols1 = sort_column1(cols1)
                    image['column_data'] = sort_cols1

                elif image['category_id'] == 2:
                    cols2 = []
                    image_bbox = image['bbox_scale']
                    division_position = (image_bbox[0] + image_bbox[2])/2
                    for pdf in pdfs:
                        pdf_bbox = pdf['bbox_scale']
                        if is_bbox_inside_image(image_bbox, pdf_bbox):
                            cols2.append(pdf)
                    sort_cols2 = sort_column2(cols2,division_position)
                    image['column_data'] = sort_cols2

                elif image['category_id'] == 4:
                    cols3 = []
                    image_bbox = image['bbox_scale']
                    division = (image_bbox[0] + image_bbox[2])/3
                    division_position1 = image_bbox[0] + division
                    division_position2 = image_bbox[0] + 2*division
                    for pdf in pdfs:
                        pdf_bbox = pdf['bbox_scale']
                        if is_bbox_inside_image(image_bbox, pdf_bbox):
                            cols3.append(pdf)
                    sort_cols3 = sort_column3(cols3,division_position1, division_position2)

                    image['column_data'] = sort_cols3
        idx+=1
    return data_png


def write_text_to_files(data):
    for sublist in tqdm(data, desc='Processing'):
        for entry in sublist:
            image_file_name = entry['image_filename']
            for column_entry in entry.get('column_data', []):
                text = column_entry.get('text', '')  # Using .get() to handle missing 'text' field
                file_path = os.path.join('data_preprocessing/txt', f"{image_file_name}.txt")  # Construct file path
                with open(file_path, 'a') as file:
                    file.write(text + '\n')

def draw_bounding_boxes_on_images(data_png, input_folder, output_folder):

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over each layout in the data
    for layout in data_png:
        # Load the image
        img_path = os.path.join(input_folder, layout[0]['image_filename'])
        image = cv2.imread(img_path)

        # Iterate over bounding boxes in the layout structure
        for bbox_info in layout:
            bbox = bbox_info['bbox_scale']
            # Scale bounding box coordinates to match the image dimensions
            image_height, image_width, _ = image.shape
            bbox_pixels = [
                int(bbox[0] * image_width),     # x1
                int(bbox[1] * image_height),    # y1
                int((bbox[2] + bbox[0]) * image_width),     # x2
                int((bbox[3] + bbox[1]) * image_height)     # y2
            ]
            # Draw bounding box on the image
            cv2.rectangle(image, (bbox_pixels[0], bbox_pixels[1]), (bbox_pixels[2], bbox_pixels[3]), (0, 255, 0), 2)
            text = f"x: {bbox_pixels[0]:.2f}, y: {bbox_pixels[1]:.2f}, width: {bbox_pixels[2]-bbox_pixels[0]:.2f}, height: {bbox_pixels[3]-bbox_pixels[1]:.2f}"
            cv2.putText(image, text, (bbox_pixels[0], bbox_pixels[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save the image with bounding boxes drawn
        filename = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(filename, image)
    
    

        
        
def main():
    FLAGS = parse_args()
    
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)
    if 'use_gpu' not in cfg:
        cfg.use_gpu = False
    if cfg.use_gpu:
        place = paddle.set_device('gpu')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_version()
    png_path = 'data_preprocessing/pages'
    pdf_path = 'data_preprocessing/pdf'
    data_png = extract_data_from_png_folder(cfg,png_path)
    
    data_pdf = extract_data_from_pdf_folder(pdf_path)
    # data_pdf = reorder_bbox_top_to_bottom(pdf_data_info)
    # final_data = process_data(data_png, data_pdf)
    # write_text_to_files(final_data)
    input_folder = 'data_preprocessing/pages'
    output_folder = 'data_preprocessing/annotatedImage'
    draw_bounding_boxes_on_images(data_png,input_folder,output_folder)
    

if __name__ == '__main__':
    main()




