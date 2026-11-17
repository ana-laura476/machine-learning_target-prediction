import pandas as pd
import numpy as np


def select_bs(raw_df):
    batch_size = 32  # default
    epochs = 500
    if raw_df.shape[0] < 250:
        batch_size = 8
        epochs = 1500
    elif 250 < raw_df.shape[0] < 500:
        batch_size = 16
        epochs = 1500
    elif 500 < raw_df.shape[0] < 4000:
        batch_size = 32
        if 500 < raw_df.shape[0] < 1500:
            epochs = 1000
        else:
            epochs = 500
    elif 4000 < raw_df.shape[0]:
        batch_size = 64
        epochs = 500

    return batch_size, epochs


