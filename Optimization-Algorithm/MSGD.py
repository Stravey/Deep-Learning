# Minibatch Stochastic Gradient Descent
# Minibatch随机梯度下降

import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

class Timer: #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):


    def avg(self):


