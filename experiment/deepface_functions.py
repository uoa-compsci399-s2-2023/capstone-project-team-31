import numpy as np
from deepface.DeepFace import *
loss_object = tf.keras.losses.CategoricalCrossentropy()
import cv2
from deepface_models import *
    
    

def get_confidence_in_true_class(image: np.ndarray, classification:str, true_class:str, e:attributeModel):
    '''
    takes a single image and returns deepface's confidence in predicting its true class
    '''
    
    image_after = image.astype(np.float32)
    image_after = np.divide(image_after, 255)
        

    labeledOutput = e.predict_verbose(image_after)
    
    confidence = labeledOutput[classification][true_class]
    
    return confidence/100
    
