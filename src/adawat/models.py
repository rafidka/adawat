import torch
from torch import nn
from typing import Any, Callable
import logging


log = logging.getLogger(__name__)


class ModelTrainer:
    """
    A class for training models. Training models contains a common
    functionality, like creating the model, creating a loss function
    and an optimizer, and so on. So, this class helps abstract the
    common functionality.

    """

    def __init__(self, model_creator: Callable[[], nn.Module],
                 loss_fn_creator: Callable[[], nn.Module],
                 optim_creator: Callable[[Any], torch.optim.Optimizer],
                 optim_updater: Callable[[torch.optim.Optimizer, int, int, int], None] = None):
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
        self.optim_updater = optim_updater

    def _train_epoch(self, model: nn.Module, epoch: int,
                     loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                     train_loader: torch.utils.data.DataLoader):
        total_loss = 0.0

        iter_count = len(train_loader)
        last_perc_completed = 0
        for i, (X, y) in enumerate(train_loader):
            if self.optim_updater is not None:
                self.optim_updater(optimizer, epoch, i, iter_count)
            model.zero_grad()

            y_train = model(X)

            loss = loss_fn(y_train, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            perc_completed = i * 100 // iter_count
            if perc_completed == 100 or perc_completed > last_perc_completed:
                log.info(
                    f"Finished {perc_completed}%. Loss is {total_loss/(i+1)}.")
                last_perc_completed = perc_completed

        return total_loss / iter_count

    def train(self, train_loader: torch.utils.data.DataLoader,
              epochs: int, device: torch.cuda.device = None):
        model = self.model_creator()
        if device is not None:
            model.to(device)
        loss_fn = self.loss_fn_creator()
        optimizer = self.optim_creator(model.parameters())

        losses = []
        for i in range(epochs):
            log.info(f"Starting epoch {i + 1}")
            loss = self._train_epoch(
                model, i, loss_fn, optimizer, train_loader)
            log.info(f"Finished epoch {i + 1}. Loss is {loss}.")
            losses.append(loss)
        return model, losses
