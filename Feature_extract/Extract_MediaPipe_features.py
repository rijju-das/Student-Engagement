import cv2
import mediapipe as mp
import itertools
import numpy as np
import os
from time import time
import pandas as pd
import matplotlib.pyplot as plt

import mediaPipeFeatureExtractor as fmp

#path to WACV image folders
path1 = "/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/dataset/1"
path2 = "/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/dataset/2"
path3 = "/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/dataset/3"


df_OF0 = pd.read_csv("/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/processedData0.csv")
df_OF1 = pd.read_csv("/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/processedData1.csv")
df_OF2 = pd.read_csv("/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/processedData2.csv")

lm_dic0 = fmp.faceMesh_extract(path1,False)
lm_dic1 = fmp.faceMesh_extract(path2,False)
lm_dic2 = fmp.faceMesh_extract(path3,False)

df_merge0 = fmp.buildFeatureDataframe(lm_dic0,0,df_OF0)
df_merge1 = fmp.buildFeatureDataframe(lm_dic1,1,df_OF1)
df_merge2 = fmp.buildFeatureDataframe(lm_dic2,2,df_OF2)


df_merge0.to_csv("/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/merged_data0.csv")
df_merge1.to_csv("/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/merged_data1.csv")
df_merge2.to_csv("/Users/rijju/Documents/GitHub/Student-Engagement/WACV data/merged_data2.csv")