import tensorflow as tf
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np

import deepface_functions as df
import deepface_models as dm
import cv2
from adversarial_pattern_generator import cleanup_dims
import image_helper_functions as imh
import random
import json

import os
os.chdir('C:/Users/matta/Desktop/Capstone/')
os.getcwd()

classification = 'ethnicity'
label = 'black'
database = 'val'
besides = True

train_df = pd.read_csv("Fairfacedb/fairface_label_train.csv")
test_df = pd.read_csv("Fairfacedb/fairface_label_val.csv")

train_df = train_df.astype({'file':str})
test_df = test_df.astype({'file':str})

train_df.rename(columns = {'race':'ethnicity'}, inplace=True)
test_df.rename(columns = {'race':'ethnicity'}, inplace=True)

train_df['emotion'] = 'neutral'
test_df['emotion'] = 'neutral'

train_df.set_index('file', inplace=True, drop=True)
test_df.set_index('file', inplace=True, drop=True)

train_df = train_df.drop(columns=['service_test'])
test_df = test_df.drop(columns=['service_test'])

if database == 'train':
    final_df = train_df
elif database == 'val':
    final_df = test_df

idx = final_df[(final_df['ethnicity'] == 'East Asian') | (final_df['ethnicity'] == 'Southeast Asian')].index
final_df.loc[idx, 'ethnicity'] = 'Asian'

if besides:
    sorted_df = final_df[final_df[classification] != label]
else: 
    sorted_df = final_df[final_df[classification] == label]
print(sorted_df)

label_json = sorted_df.to_dict(orient='index')

if besides:
    json_name = str('not' + label + '_' + database +'.json')
else:
    json_name = str('not' + label + '_' + database +'.json')

with open(json_name, "w") as outfile:
    json.dump(label_json, outfile)