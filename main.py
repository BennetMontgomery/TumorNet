import torch
from torch import nn
from tumorset import TumorSet
from tumornet import TumorNet
from train import Trainer
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# data
train_data = TumorSet('./data/train')
test_data = TumorSet('./data/test')
train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = TumorNet(basechannels=64).to(device)
trainer = Trainer(
                model=model,
                loss_fn=nn.BCELoss(),
                optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                device=device
            )
print(model)

losses = []

for epoch in range(500):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    trainer.train()
    losses.append(trainer.test())

# save model
torch.save(model.state_dict(), './checkpoints/model.pth')

plt.plot(losses)
plt.show()

print("Terminated")