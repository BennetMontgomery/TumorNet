import argparse
import torch
import warnings
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import Grayscale
from torchvision.transforms.functional import convert_image_dtype
from tumornet import TumorNet
from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Neural network architecture for semantic segmentation of brain '
                                                 'MRI scans of the FLAIR sequence type into tumor and non-tumor '
                                                 'categories.')
    parser.add_argument('--model', '-m', default='./checkpoints/model.pth', metavar='FILE', type=str,
                        help='Specify the path to a trained model file.')
    parser.add_argument('--base-channels', '-b', default=64, metavar='N', type=int,
                        help='Base channels of the model. Must match what was used for this flag during training')
    parser.add_argument('--input', '-i', default='in.png', metavar='FILE', type=str,
                        help='Specify a file name for the input image')
    parser.add_argument('--output', '-o', default='out.png', metavar='FILE', type=str,
                        help='Specify a file name for the primary output image')
    parser.add_argument('--threshold', '-t', default=0.2, metavar='FLOAT', type=float,
                        help='Specify the masking threshold for a pixel to be considered for inclusion in the output mask')
    parser.add_argument('--checkpointing', '-c', default=False, action='store_true',
                        help='Whether or not to exchange compute for memory footprint. Enables model segment checkpointing')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    model_path = args.model
    base_channels = args.base_channels
    input_path = args.input
    output_path = args.output
    threshold = args.threshold
    checkpointing = args.checkpointing

    image = read_image(input_path)
    image = Grayscale()(image)
    image = convert_image_dtype(image)

    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )

    model = TumorNet(basechannels=base_channels).to(device)
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        input_cuda = image.to(device)
        prediction = model(input_cuda, inference=True, checkpointing=checkpointing)

    prediction = np.where(np.squeeze(prediction.cpu(), axis=0) > threshold, 1, 0)

    warnings.simplefilter(action='ignore', category=UserWarning)
    plt.imshow(image.cpu().T, cmap='gray')
    plt.imshow(prediction.T, cmap='gray', alpha=0.5)
    plt.savefig(output_path)
