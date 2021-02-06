import logging

import torch

log = logging.getLogger(__name__)


def get_training_device(try_gpu=True):
    """
    Retrieves the device used for training. It favours a CUDA-enabled device,
    but reverts to CPU if it doesn't find one.

    Keyword arguments:
    try_gpu -- If true, this function will prefer a CUDA-enabled device. Otherwise,
        it will always use the CPU.

    Returns:
    The training device.
    """
    if try_gpu and torch.cuda.is_available():
        log.info("CUDA is available; using it.")
        device = torch.device("cuda:0")
    else:
        log.info("CUDA is not available; using CPU.")
        device = torch.device("cpu")
    return device
