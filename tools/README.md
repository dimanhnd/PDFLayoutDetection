<div align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" width="800" />
</div>

# PDF Layout Detection and Data Extraction

## Introduction
This repository contains Python scripts for PDF layout detection and data extraction. It utilizes the PDFMiner library for parsing PDF files, OpenCV for image processing tasks, and PaddleDetection for object detection tasks.

## Features
- **PDF Layout Detection**: Extracts text and bounding box information from PDF files.
- **Image Annotation**: Annotates images with bounding boxes representing text regions.
- **Data Extraction**: Extracts text data from PDF layouts and saves it to text files.

## Requirements
- Python 3.x
- PDFMiner
- OpenCV
- PaddleDetection
- tqdm

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dimanhnd/PDFLayoutDetection.git
   cd PDFLayoutDetection

2. If you want to experience the prediction process directly, you can skip data preparation and download the pre-trained model. Alternatively, follow the folder structure in dataset/custom_data_r to understand the data training and JSON file format.

Directory Structure:
   ```bash
   dataset/custom_data_r
   ├── train
   ├── instance_train.json
   ```
Data Annotation:
The JSON file contains annotations of all images, structured in a nested dictionary format, including information such as file name, height, width, image ID, segmentation, area, bounding box (bbox), category ID, and annotation ID.

3. Training Process
Training scripts, evaluation scripts, and prediction scripts are provided, using the PubLayNet pre-training model as an example.

First, create a directory for the pre-trained model:
``` bash
   mkdir pretrained_model
   cd pretrained_model
   wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout.pdparams
   wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar 
   ``` 
   - 3.1: Train - If you want to train your own data set, you need to modify the data configuration and the number of categories in the configuration file.
      - Model 1: Suggestion: 
        Using 'configs/picodet/legacy_model/application/layout_analysis/picodet_lcnet_x1_0_layout.yml' as an example, the change is as follows:

   ``` yaml
      metric: COCO
      # Number of categories
      num_classes: 5

      TrainDataset:
      !COCODataSet
         # Modify to your own training data directory
         image_dir: train
         # Modify to your own training data label file
         anno_path: train.json
         # Modify to your own training data root directory
         dataset_dir: /root/publaynet/
         data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

      EvalDataset:
      !COCODataSet
         # Modify to your own validation data directory
         image_dir: val
         # Modify to your own validation data label file
         anno_path: val.json
         # Modify to your own validation data root
         dataset_dir: /root/publaynet/

      TestDataset:
      !ImageFolder
         # Modify to your own test data label file
         anno_path: /root/publaynet/val.json
   ```
   - Model 2: In use: 
   Using "PDFLayoutDetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml"
   ``` yml
      # update model train
      _BASE_: [
      '../datasets/coco_detection.yml',
      '../runtime.yml',
      '_base_/optimizer_1x.yml',
      '_base_/faster_rcnn_r50_fpn.yml',
      '_base_/faster_fpn_reader.yml',
      ]
      epoch: 100
      weights: output/faster_rcnn_r50_fpn_1x_coco/model_final
   ```
   - 3.2: Start training
   ``` bash
      # GPU training supports single-card and multi-card training
      # The training log is automatically saved to the log directory

      # Single card training
      export CUDA_VISIBLE_DEVICES=0
      python3 tools/train.py \
         -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
         --eval

      # Multi-card training, with the -- GPUS parameter specifying the card number
      export CUDA_VISIBLE_DEVICES=0,1,2,3
      python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py \
         -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
         --eval
   ```

4. Model evaluation and prediction
Model parameters in training are saved by default in output/ Under the layout directory. When evaluating indicators, you need to set weights to point to the saved parameter file.Assessment datasets can be accessed via configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml.
   ``` bash
      # GPU evaluation, weights as weights to be measured
      python3 tools/eval.py \
         -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
         -o weights=./output/best_model
   ```
5. Test Layout analysis
With trained PaddleDetection model, you can use the following commands to make model predictions.
   ``` bash
      python3 tools/infer.py \
         -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
         -o weights='output/best_model.pdparams' \
         --infer_img='docs/images/layout.jpg' \
         --output_dir=output_dir/ \
         --draw_threshold=0.5
   ```
   - --infer_img: Reasoning for a single picture can also be done via --infer_ DirInform all pictures in the file.
   - --output_dir: Specify the path to save the visualization results.
   - --draw_threshold:Specify the NMS threshold for drawing the result box.

6. Using data and display and save to txt file
   ``` bash
      python3 tools/pdf_layout_detection.py \
         -c configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
         -o weights='output/best_model.pdparams' \
         --infer_img='docs/images/layout.jpg' \
         --output_dir=output_dir/ \
         --draw_threshold=0.5
   ```
7. Visualize data:
   See on output_dir folder
   ```markdown
      ![Image Name](https://github.com/dimanhnd/PDFLayoutDetection/blob/release/2.7/output_dir/page_10_page_1.png)
