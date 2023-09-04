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

    img_copy = image.astype(np.uint8)

    #cv2.imshow('img_cpoysfd', img_copy[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(np.max(image), np.min(image))

    labeledOutput = e.predict_verbose(image_after)
    print(labeledOutput)
    
    confidence = labeledOutput[classification][true_class]
    
    return labeledOutput, confidence/100
    
