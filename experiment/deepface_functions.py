import numpy as np
from deepface.DeepFace import *
loss_object = tf.keras.losses.CategoricalCrossentropy()
from deepface_models import *

def get_confidence_in_selected_class(image: np.ndarray, classification:str, true_class:str, e:attributeModel, verbose=False):
    '''
    Takes a single image and returns deepface's confidence in predicting its true class
    
    Args:
    * image: np array with shape (1, h, w, 3)
    * classification: Either ethnicity, emotion, or gender classification
    * true_class: true class of specific image
    * e: facial recognition model
    * verbose: add logging display

    Returns:
    * labeledOutput: prediction details
    * confidence: confidence score on true class
    '''
    
    image_after = image.astype(np.float32)
    image_after = np.divide(image_after, 255)

    labeledOutput = e.predict_verbose(image_after)
    if verbose:
        print(labeledOutput)
        
    confidence = labeledOutput[classification][true_class]
    
    return labeledOutput, confidence/100
    
