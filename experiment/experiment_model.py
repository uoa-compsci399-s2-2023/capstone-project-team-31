import numpy as np

# Class of image with perturbed accessory
class Experiment:
    
    def __init__(self, accessory_image, accessory_mask):
        self.__accessory_image = accessory_image
        self.__accessory_mask = accessory_mask
        
    def get_image(self):
        return np.copy(self.__accessory_image)
    
    def get_mask(self):
        return np.copy(self.__accessory_mask)
    
    def set_image(self, image):
        self.__accessory_image = image
        
    def set_mask(self, mask):
        self.__accessory_mask = mask