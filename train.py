import torch

class Trainer():
    def __init__(self, model, loss_fn, optimizer, train_dataloader, test_dataloader, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def train(self):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (features, masks) in enumerate(self.test_dataloader):
            features, masks = features.to(self.device), masks.to(self.device)

            # get model prediction mask
            prediction = self.model(features)
            loss = self.loss_fn(prediction, masks.float())

            # optimize model
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(features)
                print(f'loss: {loss:>7f}, [{current:>5d}/{size:>5d}]')

    def test(self):
        num_batches = len(self.test_dataloader)
        self.model.eval()

        test_loss = 0
        with torch.no_grad():
            for features, masks in self.test_dataloader:
                features, masks = features.to(self.device), masks.to(self.device)
                prediction = self.model(features)
                test_loss += self.loss_fn(prediction, masks.float()).item()

        test_loss /= num_batches
        print(f'evaluation loss: {test_loss:>7f}')
        return test_loss