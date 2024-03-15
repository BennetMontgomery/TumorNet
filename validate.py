import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from train import Trainer
from tumornet import TumorNet
from tumorset import TumorSet
from matplotlib import pyplot as plt

threshold = 0.14

dataset = TumorSet('./data/valid')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

model = TumorNet(basechannels=64).to(device)
model.load_state_dict(torch.load('./checkpoints/model6.pth'))

validator = Trainer(
    model=model,
    loss_fn=nn.BCEWithLogitsLoss(),
    optimizer=None,
    scheduler=None,
    train_dataloader=None,
    test_dataloader=dataloader,
    device=device
)

# validator.test(checkpointing=True)

for index in [random.randint(0, len(dataset)) for _ in range(5)]:
    validation_feature = dataset[index][0]

    with torch.no_grad():
        validation_feature_cuda = validation_feature.to(device)
        prediction = model(validation_feature_cuda, inference=True)

    prediction = np.where(np.squeeze(prediction.cpu(), axis=0) > threshold, 1, 0)

    plt.imshow(validation_feature.cpu().T, cmap='gray')
    plt.show()
    plt.imshow(prediction.T, cmap='gray')
    plt.show()

    plt.imshow(validation_feature.cpu().T, cmap='gray')
    plt.imshow(prediction.T, cmap='gray', alpha=0.5)
    plt.show()