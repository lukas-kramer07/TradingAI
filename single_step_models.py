## A number of single step models, prediciting one day into the future based on the last day of data

#imports
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from WindowGen import WindowGenerator
from dataengineering import return_data




def main():
    #get data
    train_df, val_df, test_df, column_indices, num_features = return_data()
    


if __name__ == '__main__':
    main()