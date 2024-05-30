# Advanced LSTM model specialized on prediciting stock prices

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from single_step_models import ResidualWrapper
from utils import WindowGenerator
from utils import concat_data, compile_and_fit, plot
from multi_step_models import PERFORMANCE, VAL_PERFORMANCE, LastStepBaseline
import os

VAL_PERFORMANCE = {}
PERFORMANCE = {}

def main():
    pass
def test(model, window, name):
    VAL_PERFORMANCE[name] = model.evaluate(window.val, return_dict=True)
    PERFORMANCE[name] = model.evaluate(window.test, verbose=0, return_dict=True)



if __name__ == '__main__':
    main()