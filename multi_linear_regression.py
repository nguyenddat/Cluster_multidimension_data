import torch 
import numpy as np
import pandas as pd
import torch.nn as nn

"""
Multiple Linear Regression

Model 2 lớp Linear Regression nhận dữ liệu có:
- Đầu vào: time
- Đầu ra: dữ liệu
"""

def time_encoding(df, period = 24):
    df["start_hour"] = df["start_time"].dt.hour + df["start_time"].dt.minute / 60
    df["end_hour"] = df["end_time"].dt.hour + df["end_time"].dt.minute / 60

    starts = np.array(df["start_hour"])
    ends = np.array(df["end_hour"])

    sin_s = np.sin(2 * np.pi * starts / period) 
    cos_s = np.cos(2 * np.pi * starts / period) 
    
    sin_e = np.sin(2 * np.pi * ends / period)
    cos_e = np.cos(2 * np.pi * ends / period)

    return np.stack([sin_s, cos_s, sin_e, cos_e], axis = 1) 


class MultiLinearRegression(nn.Module):
    def __init__(self):
        super(MultiLinearRegression, self).__init__()

    def read_df(self, time: pd.DataFrame, value: pd.DataFrame):
        X = time_encoding(time)
        y = np.array(value)

        return X, y
    

multi_lr = MultiLinearRegression()
