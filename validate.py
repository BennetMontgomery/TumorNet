import torch
import random
import logging
import argparse
import warnings
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from train import Trainer
from tumornet import TumorNet
from tumorset import TumorSet
from matplotlib import pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('Validator script for TumorNet. Used to calculate loss, accuracy, precision, '
                                     'recall, and f-score of a trained TumorNet model. Optionally outputs visual'
                                     ' validation.')
    parser.add_argument('--model', '-m', default='./checkpoints/model.pth', metavar='PATH', type=str,
                        help='Path to trained instance of TumorNet model')
    parser.add_argument('--base-channels', '-bc', default=64, metavar='N', type=int,
                        help='Base channels argument used to train model')
    parser.add_argument('--valid-data', '-vd', default='./data/valid', metavar='PATH', type=str,
                        help='Path to validation dataset')
    parser.add_argument('--threshold', '-t', default=0.20, metavar='FLOAT', type=float,
                        help='Threshold to consider tissue tumor tissue for validation purposes, model dependent')
    parser.add_argument('--batch-size', '-b', default=2, metavar='N', type=int,
                        help='Batch size for model evaluation. Should be a power of 2')
    parser.add_argument('--checkpointing', '-c', default=False, action='store_true',
                        help='Whether or not to exchange compute for memory footprint. Enables model segment checkpointing')
    parser.add_argument('--visualize', '-v', default=0, metavar='Z+', type=int,
                        help='If included, outputs Z+ (integer >= 0) visual validation examples')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    model_path = args.model
    validation_data = TumorSet(args.valid_data)
    base_channels = args.base_channels
    threshold = args.threshold
    batch_size = args.batch_size
    checkpointing = args.checkpointing
    visualize = args.visualize

    dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )

    model = TumorNet(basechannels=base_channels).to(device)
    model.load_state_dict(torch.load(model_path))

    validator = Trainer(
        model=model,
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer=None,
        scheduler=None,
        train_dataloader=None,
        test_dataloader=dataloader,
        device=device
    )

    try:
        loss, acc, precision, recall, fscore = validator.validate(threshold=threshold, checkpointing=checkpointing)
    except torch.cuda.OutOfMemoryError:
        logging.error('Memory usage exceeded! Enable checkpointing mode with --checkpointing or -c')
        exit(1)

    print(f'accuracy | precision | recall | fscore: \n{acc:.2f}%   | {precision:.2f}%    | {recall:.2f}% | {fscore/100:.4f}')

    if visualize > 0:
        for index in [random.randint(0, len(validation_data)) for _ in range(visualize)]:
            validation_feature = validation_data[index][0]

            with torch.no_grad():
                validation_feature_cuda = validation_feature.to(device)
                prediction = model(validation_feature_cuda, inference=True)

            prediction = np.where(np.squeeze(prediction.cpu(), axis=0) > threshold, 1, 0)

            warnings.simplefilter(action='ignore', category=UserWarning)
            plt.imshow(validation_feature.cpu().T, cmap='gray')
            plt.imshow(prediction.T, cmap='gray', alpha=0.5)
            plt.show()