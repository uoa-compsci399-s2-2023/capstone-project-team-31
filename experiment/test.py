""" import numpy as np
from image_helper_functions import *
from PIL import Image """

import tensorflow as tf
import deepface_functions as df
import deepface_models as dm
import cv2
from adversarial_pattern_generator import cleanup_dims
import image_helper_functions as imh
#print(tf.reduce_sum(tf.random.normal([1000, 1000])))
#print(tf.config.list_physical_devices('GPU'))

e = dm.attributeModel('gender')
img = cv2.imread('experiment/test_images/pls.jpg')
temp = [img, 'Asian', 'male', '23', 'neutral']
prep_img = imh.image_to_face(temp)
print(prep_img)

final_img = prep_img[0]
cv2.imshow('image',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(e.predict_verbose(cleanup_dims(final_img)))

""" test = np.random.rand(4,4,3)
test = test/np.max(np.abs(test))
seg = np.ones((224,224,3))
image = Image.open('test.jpg')
#huh = get_printable_vals(4)
#print(seg)
scores = np.ones((test.shape[0], test.shape[1]))
gradient = np.zeros(test.shape)

arr = []
with open('experiment/assets/printable_vals.txt') as file:
    lines = file.readlines()
    for line in lines:
        line = line.split()
        line = list(map(int, line))
        arr.append(line)
    print(arr) """

#print(non_printability_score(test,seg,huh))

""" image = Image.open('experiment/assets/test.jpg')
img = np.array(image)
img = img.astype(np.float64) """

#print(non_printability_score(img, seg, nps))

