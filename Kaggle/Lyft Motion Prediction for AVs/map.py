# Understanding the data + Catalyst/Kekas baseline
import numpy as np
import l5kit, os
import matplotlib.pyplot as plt
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "./input/lyft-motion-prediction-autonomous-vehicles"
# get config
cfg = load_config_data("../input/lyft-config-files/visualisation_config.yaml")