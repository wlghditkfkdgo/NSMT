from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

import os
from sys import stdout

# snn
from spikingjelly.clock_driven import functional
from syops import get_model_complexity_info

# model
from timm import create_model
from timm.optim.optim_factory import create_optimizer_v2

from config import set_random_seed, parse_arguments, Config
from utils import get_energy_consumption
import model as model
from ours import TemporalBlock
from data_provider.data_factory import data_provider
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from analysis import *

# model loading
