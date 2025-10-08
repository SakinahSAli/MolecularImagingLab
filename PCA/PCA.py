from sklearn.datasets import load_digits
import pandas as pd

#load the digits dataset
dataset = load_digits()
# the returned object is a Bunch with .data (n_samples, n_features)
print(dataset.data.shape)


