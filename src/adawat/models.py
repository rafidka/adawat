import logging
from operator import itemgetter
from typing import Any, Callable

import torch
from torch import nn
from adawat.picklers import MemoryPickler

from adawat.serialization import stateful
from adawat.state_machine import PersistentStateMachine

log = logging.getLogger(__name__)


class ModelTrainerStateMachine(PersistentStateMachine):
    def __init__(
            self,
            id: str,
            model_creator: Callable[[], nn.Module],
            train_loader: torch.utils.data.DataLoader,
            epochs: int,
            loss_fn_creator: Callable[[], nn.Module],
            optim_creator: Callable[[Any], torch.optim.Optimizer],
            optim_updater: Callable[[
                torch.optim.Optimizer, int, int, int], None] = None,
            device: torch.cuda.device = None,
            pickler=MemoryPickler()):
        """
        model_creator -- A function with no parameters that when called creates
            the model.

        train_loader -- The training data loader.

        epochs -- The number of training epochs.

        loss_fn_creator -- A function with no parameters that when called
            create the loss function. An example:

            ```
            def loss_fn_creator():
                return nn.NLLLoss()
            ```

        optim_creator -- A function that creates the optimizer for training the
            model. It takes the model parameters as the first and only
            parameter. It returns the optimizer. Below is an example:

            ```
            def optim_creator(parameters):
                return optim.SGD(parameters, 0.001)
            ```

        optim_updater -- (Optional) A function that updates the optimizer used
            for training. It takes the optimizer, training epoch, training
            iteration, and training total iterations as parameters, in that
            order. Below is an example which gradually reduces the learning
            rate of the optimizer defined above.:

            ```
            lr_start = 0.001
            lr_end = 0.00001

            def optim_updater(optim, epoch, iter, iter_count):
                # Find the absolute index of the iteration, i.e. considering
                # iterations from all training epochs.
                abs_index = epoch*iter_count + i

                # Find the index of the last iteration in the training session,
                # i.e. considering all training epochs.
                last_index = epoch_count*iter_count - 1

                # Calculate the new learning rate.
                new_lr = lr_start + (lr_end - lr_start)*abs_index/last_index

                print(f'Changing learning rate to {new_lr}')
                for g in optim.param_groups:
                    g['lr'] = new_lr
            ```
        """
        super().__init__(id, "train_init", pickler)  # train_init is the starting state
        self.model_creator = model_creator
        self.loss_fn_creator = loss_fn_creator
        self.train_loader = train_loader
        self.epochs = epochs
        self.optim_creator = optim_creator
        self.optim_updater = optim_updater
        self.model = None
        self.device = device

    def train_init(self):
        # Create the necessary stuff for training.
        self.model = self.model_creator()
        loss_fn = self.loss_fn_creator()
        optim = self.optim_creator(self.model.parameters())

        # If the user specified a device, move the model to it.
        if self.device is not None:
            self.model.to(self.device)

        return self.train_epoch.__name__, {
            "model_state_dict": self.model.state_dict(),
            "loss_fn": loss_fn,
            "optim": optim,
            "epoch": 1,
            "epochs": self.epochs,
            "losses": []
        }

    def train_epoch(self):
        model_state_dict, loss_fn, optim, epoch, epochs, losses = itemgetter(
            "model_state_dict", "loss_fn", "optim", "epoch", "epochs", "losses")(self.state.data)

        # Create a model and load the latest state dictionary.
        self.model = self.model_creator()
        self.model.load_state_dict(model_state_dict)

        # If the user specified a device, move the model to it.
        if self.device is not None:
            self.model.to(self.device)

        total_loss = 0.0

        iter_count = len(self.train_loader)
        last_perc_completed = 0
        for i, (X, y) in enumerate(self.train_loader):
            if self.optim_updater is not None:
                self.optim_updater(optim, epoch, i, iter_count)
            self.model.zero_grad()

            y_train = self.model(X)

            loss = loss_fn(y_train, y)

            loss.backward()
            optim.step()

            total_loss += loss.item()

            perc_completed = i * 100 // iter_count
            if perc_completed == 100 or perc_completed > last_perc_completed:
                log.info(
                    f"Finished {perc_completed}%. Loss is {total_loss/(i+1)}.")
                last_perc_completed = perc_completed

        losses.append(total_loss / iter_count)

        if epoch <= epochs:
            # Still more epochs.
            return self.train_epoch.__name__, {
                "model_state_dict": self.model.state_dict(),
                "loss_fn": loss_fn,
                "optim": optim,
                "epoch": epoch + 1,
                "epochs": epochs,
                "losses": losses
            }
        else:
            # Finished training.
            return None, {}
