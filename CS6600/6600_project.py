import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from google.colab import drive

df = fetch_openml(data_id=43496, parser="pandas", as_frame=True).frame
df.info()
df.head()