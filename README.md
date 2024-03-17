# TumorNet

Files for training a neural network architecture for semantic segmentation of brain MRI scans of the FLAIR sequence type
into tumor and non-tumor categories. Based on the [U-Net architecture](https://arxiv.org/pdf/1505.04597.pdf) proposed by
Ronneberger et al., 2015. Trained using the [Brain Tumor Image Dataset](https://gts.ai/dataset-download/brain-tumor-image-dataset-semantic-segmentation/).

## Usage
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

To validate a trained model, run ``python validate.py``. Ensure line 25 points to the correct model. 

A dataset of annotated brain tumor images must be placed in a ``./data`` folder, separated into ``./data/train``, 
``./data/test``, and ``./data/valid`` in order for the training and validation scripts to work out of the box. 

During inference and training, images must pre-scaled to 640x640. 

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
* Scripts for easily customizable validation and prediction
* A script for easy dataset fetching
* Automatic input scaling to match expected image size
