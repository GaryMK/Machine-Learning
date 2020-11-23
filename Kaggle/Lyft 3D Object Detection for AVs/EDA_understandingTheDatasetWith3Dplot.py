import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
from IPython.display import HTML

import pdb
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

# Load the SDK
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

from moviepy.editor import ImageSequenceClip
from tqdm import tqdm_notebook as tqdm

lyftdata = LyftDataset(data_path='./input/3d-object-detection-for-autonomous-vehicles/', json_path='./input/3d-object-detection-for-autonomous-vehicles/data/', verbose=True)

print(lyftdata.category[0])
cat_token = lyftdata.category[0]['token']
print(cat_token)
lyftdata.get('category', cat_token)
print(lyftdata.sample_annotation[0])
print(lyftdata.get('attribute', lyftdata.sample_annotation[0]['attribute_tokens'][0]))

# Scenes
print(lyftdata.scene[0])
train = pd.read_csv('./input/3d-object-detection-for-autonomous-vehicles/train.csv')
print(train.head())
token0 = train.iloc[0]['Id']
print(token0)

# Sample
my_sample = lyftdata.get('sample', token0)
print(my_sample)


# 3D interactive visualization of a sample
lyftdata.render_sample_3d_interactive(my_sample['token'], render_sample=False)
print(my_sample.keys())
print(lyftdata.sensor)
sensor = 'CAM_FRONT'
cam_front = lyftdata.get('sample_data', my_sample['data'][sensor])
print(cam_front)
img = Image.open(cam_front['filename'])
print(img)