import matplotlib.pyplot as plt
import pandas as pd
from data_prep import load_and_clean
from model_linear import train_linear
from model_lasso import train_lasso
from model_rf import train_rf
from model_gb import train_gb