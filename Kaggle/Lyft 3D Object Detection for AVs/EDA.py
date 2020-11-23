import os
import pandas as pd
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import warnings

plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
warnings.filterwarnings('ignore')
pd.options.display.max_columns = 99
sns.set_palette(sns.color_palette('tab20', 20))

start = dt.datetime.now()

base = './input/3d-object-detection-for-autonomous-vehicles/'
dirs = os.listdir(base)
print(dirs)

train = pd.read_csv(base + 'train.csv')
sample_submission = pd.read_csv(base + 'sample_submission.csv')
print(f'train: {train.shape}, sample submission: {sample_submission.shape}')
print(train.head(2))
print(sample_submission.head(2))

# check the parsing of prediction strings
max([len(ps.split(' ')) % 8 for ps in train.PredictionString.values])


