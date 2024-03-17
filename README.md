# TumorNet

Files for training a neural network architecture for semantic segmentation of brain MRI scans of the FLAIR sequence type
into tumor and non-tumor categories. Based on the [U-Net architecture](https://arxiv.org/pdf/1505.04597.pdf) proposed by
Ronneberger et al., 2015. Trained using the [Brain Tumor Image Dataset](https://gts.ai/dataset-download/brain-tumor-image-dataset-semantic-segmentation/).

## Usage
Three scripts, ``train_model.py``, ``validate.py``, and ``predict.py`` are included for easy training, validation, and
inference of custom models. During training, validation, and inference images must pre-scaled to 640x640. 
### Training
To train a custom model, run \
``python train_model.py``\
Training will proceed with a default list of assumptions. To customize training, use the following command line options:
```
  -h, --help            show this help message and exit
  --train-data PATH, -tr PATH
                        Location of training data. Must include a
                        _annotations.coco.json file.
  --test-data PATH, -te PATH
                        Location of test data. Must include a
                        _annotations.coco.json file.
  --model-path PATH, -m PATH
                        Location to save model when training is complete
  --checkpointing, -c   Whether or not to exchange compute for memory
                        footprint. Enables model segment checkpointing
  --batch-size N, -bs N
                        Batch size of training and testing data. Should be a
                        power of 2
  --base-channels N, -bc N
                        Number of channels in the first and last model layer.
                        Determines parameter count. Should be a multiple of 16
  --learning-rate N, -lr N
                        Learning rate of the model optimizer. Should be 10 >=
                        0.0001, <= 0.1
  --epochs N, -e N      Number of epochs to train the model
 ```

The default flags are
```
--train-data ./data/train --test-data ./data/test --model-path ./checkpoints/model.pth --batch-size 2 --base-channels 64 \
--learning-rate 0.001 --epochs 5 
```

### Validation
To validate a trained model, run ``python validate.py``. Validation will proceed with a default list of assumptions. To
customize validation, use the following command line options:
```
-h, --help            show this help message and exit
--model PATH, -m PATH
                    Path to trained instance of TumorNet model
--base-channels N, -bc N
                    Base channels argument used to train model
--valid-data PATH, -vd PATH
                    Path to validation dataset
--threshold FLOAT, -t FLOAT
                    Threshold to consider tissue tumor tissue for
                    validation purposes, model dependent
--batch-size N, -b N  Batch size for model evaluation. Should be a power of
                    2
--checkpointing, -c   Whether or not to exchange compute for memory
                    footprint. Enables model segment checkpointing
--visualize Z+, -v Z+
                    If included, outputs Z+ (integer >= 0) visual
                    validation examples
```
The following flags are default:
```
--model ./checkpoints/model.pth --base-channels 64 --valid-path ./data/valid --threshold 0.20 --batch-size 2
```
Ensure ``--base-channels`` matches the ``--base-channels`` option used for training.

### Inference
In order to use a trained model on an arbitrary input image, run ``python predict.py``. An output image will be 
generated overlaying a mask on the input image. To customize the inference process, use the following command line
options:
```
-h, --help            show this help message and exit
--model FILE, -m FILE
                    Specify the path to a trained model file.
--base-channels N, -b N
                    Base channels of the model. Must match what was used
                    for this flag during training
--input FILE, -i FILE
                    Specify a file name for the input image
--output FILE, -o FILE
                    Specify a file name for the primary output image
--threshold FLOAT, -t FLOAT
                    Specify the masking threshold for a pixel to be
                    considered for inclusion in the output mask
--checkpointing, -c   Whether or not to exchange compute for memory
                    footprint. Enables model segment checkpointing
```
The following flags are default:
```
--model ./checkpoints/model.pth --base-channels 64 --input in.png --output out.png --threshold 0.20
```
As with validation, ensure ``--base-channels`` matches the ``--base-channels`` option used for training.

## Installation
To install TumorNet, 
1. download this repository with \
``git clone https://github.com/BennetMontgomery/TumorNet.git``
2. Download the dataset from [GTS](https://gts.ai/dataset-download/brain-tumor-image-dataset-semantic-segmentation/) or [Kaggle](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation)
3. Extract dataset contents into the same folder as the repository with the following structure: \
``.``\
``|-data``\
``__|-train``\
``__|-test``\
``__|-valid``\
Each folder should contain a set of images and a ``_annotations.coco.json`` file provided by the dataset.
4. Install requirements with ``pip install pytorch matplotlib numpy``

## Requirements
Presently, the latest versions of ``pytorch``, ``matplotlib``, and ``numpy`` are required to run this project.

## Features
TumorNet is able to segment MRI images of human brains from saggital, axial, and coronal angles, identifying tumor 
tissue with up to 99% accuracy. TumorNet takes in a batch of MRI images and generates a set of masks which, when 
appropriately thresholded and drawn over the input images, identify potentially problematic tissue regions. 

Example network output on an MRI image along the coronal axis containing a pituitary tumor:
![Input image](https://i.imgur.com/03k8bSA.png)
![Output mask](https://i.imgur.com/sRTTzSG.png)
![Output mask overlaid on input image, alpha=0.5](https://i.imgur.com/KDX4IxW.png) \
The first image is the input image, the second is the output mask, and the third image is the output mask overlaid on 
input image to highlight the predicted tumor location. 

### Limitations
TumorNet was developed as a hobby project and is not intended for use as-is in a clinical setting. Please do not attempt
to use this model to diagnose a real person without my permission. While TumorNet has a high accuracy, precision and
recall are low (54% and 47% respectively). The model frequently fails to identify specific tumor types. If tumor tissue appears as dark on the MRI
image, the model may fail to detect it. Occasionally, the model will tag soft tissue outside the brain, such as adipose
tissue, as tumor tissue. 

## Planned Features
The following features are planned for the immediate future:
* A script for easy dataset fetching and requirements setup
* Automatic input scaling to match expected image size
* Link to a pretrained model for download
