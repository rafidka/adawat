import torch
from torch import nn
from typing import Any, Callable


class ModelTrainer:
    """
    A class for training models. Training models contains a common
    functionality, like creating the model, creating a loss function
    and an optimizer, and so on. So, this class helps abstract the
    common functionality.

    """

    def __init__(self, model_creator: Callable[[], nn.Module],
                 loss_fn_creator: Callable[[], nn.Module],
                 optim_creator: Callable[[Any], torch.optim.Optimizer]):
        """
        model_creator -- A function with no parameters that when called create
                         the model.
        loss_fn_creator -- A function with no parameters that when called
                           create the loss function.
        optim_creator -- A function that takes the model parameters as the
                         first and only parameter and return an optimizer for
                         training the model.
        """
        self.model_creator = model_creator
        self.loss_fn_creator = loss_fn_creator
        self.optim_creator = optim_creator

    def _train_epoch(self, model: nn.Module, loss_fn: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     train_loader: torch.utils.data.DataLoader):
        total_loss = 0.0

        for X, y in train_loader:
            y_train = model(X)

            loss = loss_fn(y_train, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss

    def train(self, train_loader: torch.utils.data.DataLoader,
              epochs: int, device: torch.cuda.device):
        model = self.model_creator()
        loss_fn = self.loss_fn_creator()
        optimizer = self.optim_creator(model.parameters())

        losses = []
        for _ in range(epochs):
            losses.append(self._train_epoch(
                model, loss_fn, optimizer, train_loader))
        return model, losses
