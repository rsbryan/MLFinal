import os
import numpy as np 

import torch
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Input_Visualization import load_data, preprocess_data

class NNHousing(nn.Module):
    def __init__(self, input_dim):
        super(NNHousing, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

##load data
def load_data():
    #load and use preprocessing from input visualizations
    df = load_data("realtor-data.zip.csv")
    df = preprocess_data(df)

    # select columns
    X = df[["bed","bath", "house_size" ]]



    
def train_loop(dataloader, model, loss_fn, optimizer, verbose=True):
    for i, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        ### START YOUR CODE ###
        pred = model(X) # Get the prediction output from model
        loss = loss_fn(pred, y) # compute loss by calling loss_fn()
        ### END YOUR CODE ###

        # Backpropagation
        ### START YOUR CODE ###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ### END YOUR CODE ###

        if verbose and i % 100 == 0:
            loss = loss.item()
            current_step = i * len(X)
            print(f"loss: {loss:>7f}  [{current_step:>5d}/{len(dataloader.dataset):>5d}]")