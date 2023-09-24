import cv2, os, json
import numpy as np
from image_helper_functions import *
from deepface_functions import *
from deepface_models import *



def cleanup_dims(image):
    
    ## cleaning up dimension issues:
    if len(np.shape(image)) == 3:
        image = np.expand_dims(image, axis=0)
        
    return image


image_dir = "D:/Github/capstone-project-team-31/database_functions/images/"
images = os.listdir(image_dir)

color_acc, mask = prepare_accessory('yellow', "D:/Github/capstone-project-team-31/experiment/assets/facemask.png", 'facemask')

deep_face_model = attributeModel('gender')

males = 0
females = 0

predicted_male = 0
predicted_female = 0


with open("D:/Github/capstone-project-team-31/Faces.json", 'r') as f:
    data = json.load(f)
    available_images = list(data.keys())
    for image in available_images:
        label = ""
        if image in images:
            if data[image]['gender'] == "Male":
                males += 1
                label = "Man"
            else:
                females += 1
                label = "Woman"
            img = cv2.imread(image_dir + image)
            prepared_image = getImageContents(img)
            
            if(prepared_image != None):
                prepared_image = np.multiply(prepared_image[0], 255).astype(np.uint8)[0]
                mask_applied = apply_accessory(prepared_image, mask, mask)

                mask_applied = np.expand_dims(mask_applied, axis=0)
                mask_applied = mask_applied.astype(np.float32)
                mask_applied = np.divide(mask_applied, 255)
                prediction = deep_face_model.predict_verbose(mask_applied)
                if prediction['dominant_gender'] == label:
                    print("predict correct: ", prediction['dominant_gender'])
                    if label == "Man":
                        predicted_male += 1
                    else:
                        predicted_female += 1
                else:
                    print("predict wrong: ", prediction['dominant_gender'])
                    
print("Accuracy on predicting male: ", predicted_male / males)
print("Accuracy on predicting female: ", predicted_female / females)