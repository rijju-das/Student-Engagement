# -*- coding: utf-8 -*-
""" File for generating Media Pipe facial features

Original file is located at
    https://colab.research.google.com/drive/1XbJjVsUptvBHMPqyvauLjEBnFyRRz5L4
"""

import cv2
import mediapipe as mp
import itertools
import numpy as np
import os
from time import time
import pandas as pd
import matplotlib.pyplot as plt


def faceMesh_extract(file_path,draw=True):
    g= globals()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                             min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    lm_dic= {}
    for file in os.listdir(file_path):
        image = cv2.imread(os.path.join(file_path,file))
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh_images.process(imgRGB)
        lm_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw == True:
                    mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_CONTOURS, mp_drawing_spec, mp_drawing_spec)
                for id, lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = image.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    lm_list+=[x,y]
        img_id = file.partition('.')[0]
        lm_dic[img_id]=lm_list
    return lm_dic

def buildFeatureDataframe(landmark, label, df_OF):
    df = pd.DataFrame(landmark.items(), columns=["ImageID", "landmarks"])
    col_name = []
    for i in range(0,468):
      col_name.append("x"+str(i))
      col_name.append("y"+str(i))
    df_split = pd.DataFrame(df["landmarks"].tolist(), columns=col_name)
    df_split = df_split.fillna(-1)
    df_split = df_split.astype('int')

    df = pd.concat([df, df_split], axis=1)

    #drop any row with nan value
    df = df[df["x0"] != -1]
    df = df.drop('landmarks', axis=1)

    #add the label column
    label_col = [label]* len(df.index)
    df["Label"] = label_col  

    #merge both the openFace dataframe and mediapipe dataframe
    df_merge = pd.merge(df, df_OF, how='inner', on = 'ImageID')

    return df_merge