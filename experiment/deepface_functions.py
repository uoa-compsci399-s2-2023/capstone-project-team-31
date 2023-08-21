import numpy as np
from deepface.DeepFace import *
loss_object = tf.keras.losses.CategoricalCrossentropy()
import cv2
from deepface_models import *

# def find_gradient_outer(image, true_class):
#     '''
#     returns deepface's confidence in the true classes and gradients for the images
#     '''
#     # inspo: https://github.com/mahmoods01/accessorize-to-a-crime/blob/master/aux/attack/find_gradient.m

#     # main difference: we want to dodge gender and age true class classification, not identification
    
#     true_class = cleanup_labels(true_class)
    
#     find_gradient

#     pass


    


def get_confidence_in_true_class(image: np.ndarray, classification:str, true_class:str, e:attributeModel):
    '''
    takes a single image and returns deepface's confidence in predicting its true class
    '''

    ## cleaning up dimension issues:
    if len(np.shape(image)) == 3:
        image = np.expand_dims(image, axis=0)
    
    
    image_after = image.astype(np.float32)
    image_after = np.divide(image_after, 255)
        

    labeledOutput = e.predict_verbose(image_after)
    print(labeledOutput)
    
    confidence = labeledOutput[classification][true_class]
    
    return confidence/100
    