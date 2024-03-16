import torch

class Trainer():
    def __init__(self, model, loss_fn, optimizer, scheduler, train_dataloader, test_dataloader, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device

    def train(self, checkpointing=False):
        size = len(self.train_dataloader.dataset)
        self.model.train()
        for batch, (features, masks) in enumerate(self.train_dataloader):
            features, masks = features.to(self.device), masks.to(self.device)

            # get model prediction mask
            prediction = self.model(features, checkpointing=checkpointing)

            loss = self.loss_fn(prediction, masks.float())

            # optimize model
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(features)
                print(f'loss: {loss:>7f}, [{current:>5d}/{size:>5d}]')

    def test(self, checkpointing=False):
        num_batches = len(self.test_dataloader)
        self.model.eval()

        test_loss = 0
        with torch.no_grad():
            for features, masks in self.test_dataloader:
                features, masks = features.to(self.device), masks.to(self.device)
                prediction = self.model(features, checkpointing=checkpointing)
                test_loss += self.loss_fn(prediction, masks.float()).item()

        test_loss /= num_batches

        if self.scheduler is not None:
            self.scheduler.step(test_loss)

        return test_loss

    def validate(self, threshold, calculate_loss=False, checkpointing=False):
        # determine loss
        loss = self.test(checkpointing=checkpointing) if calculate_loss else None

        size = len(self.test_dataloader.dataset)
        # determine accuracy, precision, recall
        accuracy = 0
        precision = 0
        recall = 0
        num_batches = len(self.test_dataloader)
        self.model.eval()

        with torch.no_grad():
            for features, masks in self.test_dataloader:
                features, masks = features.to(self.device), masks.to(self.device)
                prediction = self.model(features, inference=True, checkpointing=checkpointing)
                thresholded_prediction = (prediction > threshold).float()
                accuracy += 100 * ((thresholded_prediction == masks).float().sum()/masks.numel())

                # calculate in-place precision
                true_pos = thresholded_prediction.logical_and(masks).float().sum()/masks.numel()
                if thresholded_prediction.float().sum() != 0:
                    precision += 100 * (true_pos / (thresholded_prediction.float().sum()/thresholded_prediction.numel()))

                # calculate in-place recall
                if masks.float().sum()/masks.numel() != 0:
                    recall += 100 * (true_pos / (masks.float().sum()/masks.numel()))

            # average out accuracy, precision, and recall
            accuracy = accuracy/num_batches
            precision = precision/num_batches
            recall = recall/num_batches

        # calculate f-score from precision and recall
        fscore = 2 * (precision * recall) / (precision + recall) if precision != 0 and recall != 0 else 0

        return loss, accuracy, precision, recall, fscore