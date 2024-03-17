import logging
import os
import argparse
import torch
from torch import nn
from tumorset import TumorSet
from tumornet import TumorNet
from train import Trainer
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # debugging environmental variable to provide more error context on nvidia chips


def get_args():
    parser = argparse.ArgumentParser(description='Training script for TumorNet model. Used to generate a model with a'
                                                 ' custom set of user-defined hyperparameters, training data, and output'
                                                 ' model destination.')
    parser.add_argument('--train-data', '-tr', type=str, metavar='PATH', default='./data/train',
                        help='Location of training data. Must include a _annotations.coco.json file.')
    parser.add_argument('--test-data', '-te', type=str, metavar='PATH', default='./data/test',
                        help='Location of test data. Must include a _annotations.coco.json file.')
    parser.add_argument('--model-path', '-m', type=str, metavar='PATH', default='./checkpoints/model.pth',
                        help='Location to save model when training is complete')
    parser.add_argument('--checkpointing', '-c', action='store_true', default=False,
                        help='Whether or not to exchange compute for memory footprint. Enables model segment checkpointing')
    parser.add_argument('--batch-size', '-bs', type=int, default=2, metavar='N',
                        help='Batch size of training and testing data. Should be a power of 2')
    parser.add_argument('--base-channels', '-bc', type=int, default=64, metavar='N',
                        help='Number of channels in the first and last model layer. Determines parameter count. Should be a multiple of 16')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, metavar='N',
                        help='Learning rate of the model optimizer. Should be 10 >= 0.0001, <= 0.1')
    parser.add_argument('--epochs', '-e', type=int, default=5, metavar='N',
                        help='Number of epochs to train the model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    train_data = TumorSet(args.train_data)
    test_data = TumorSet(args.test_data)
    model_path = args.model_path
    checkpointing = args.checkpointing
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    base_channels = args.base_channels
    epochs = args.epochs


    # DataLoader classes. train_loader batchifies training data directory and test_loader batchifies testing data
    # directory.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # load to GPU/TPU device
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else
        'cpu'
    )

    # allocate model
    model = TumorNet(basechannels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    trainer = Trainer(
        model=model,
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        device=device
    )

    losses = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        try:
            trainer.train(checkpointing=checkpointing)
            test_loss = trainer.test(checkpointing=checkpointing)
        except torch.cuda.OutOfMemoryError:
            logging.error('Memory usage exceeded! Enable checkpointing mode with --checkpointing or -c')
            exit(1)


        losses.append(test_loss)
        print(f'evaluation loss: {test_loss:>7f}')
        if losses[-1] == min(losses):
            # save model if it beats best test performance in this epoch
            torch.save(model.state_dict(), model_path)


    plt.plot(losses)
    plt.show()

    print(f'Terminated successfully after {epochs} epochs')