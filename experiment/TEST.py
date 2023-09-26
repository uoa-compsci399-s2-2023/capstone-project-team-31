import cv2, os, json
import numpy as np
from image_helper_functions import *
from deepface_functions import *
from deepface_models import *
import pandas as pd


def cleanup_dims(image):
    
    ## cleaning up dimension issues:
    if len(np.shape(image)) == 3:
        image = np.expand_dims(image, axis=0)
        
    return image

colours = ["white", "black"]
accessories = ["facemask", "bandana", "glasses"] 

image_dir = "D:/Fairface/"
images = os.listdir(image_dir)



deep_face_model = attributeModel('gender')

males = 0
females = 0

predicted_male = 0
predicted_female = 0

ethnicities = ["White", "Black", "East Asian"]

# image_label_dir = "D:/Github/capstone-project-team-31/Faces.json"
image_label_dir = "D:/Fairface/train_labels.csv"

def json_eval(males, females, predicted_male, predicted_female):
    with open(image_label_dir, 'r') as f:
        data = json.load(f)
        available_images = list(data.keys())
        
        for image in available_images:
            label = ""
            if image in images:
                if data[image]['gender'] == "Male" and data[image]['ethnicity'] == ethnicity:
                    if males < 1000:
                        males += 1
                        label = "Man"
                    else:
                        continue
                elif data[image]['gender'] == "Female" and data[image]['ethnicity'] == ethnicity:
                    if females < 1000:
                        females += 1
                        label = "Woman"
                    else:
                        continue
                else:
                    continue
                img = cv2.imread(image_dir + image)
                
                try:
                    prepared_image = getImageContents(img)
                except:
                    continue
                
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
    
def fairface_eval(males, females, predicted_male, predicted_female):
    df = pd.read_csv(image_label_dir)
    gender = "Male"
    male_images = df.loc[df.gender.eq(gender) & df.race.eq(ethnicity)].sample(1000).index
    gender = "Female"
    female_images = df.loc[df.gender.eq(gender) & df.race.eq(ethnicity)].sample(100).index
    
    # male_images = df.loc[ df.race.eq(ethnicity)].sample(1000).index
    label = "Man"
    for i in male_images:
        if males < 1000:
            prepared_image = cv2.imread(image_dir + df.iloc[i]['file'])
            
            males += 1
            # prepared_image = np.multiply(prepared_image[0], 255).astype(np.uint8)[0]
            mask_applied = apply_accessory(prepared_image, color_acc, mask)
            
            mask_applied = np.expand_dims(mask_applied, axis=0)
            mask_applied = mask_applied.astype(np.float32)
            mask_applied = np.divide(mask_applied, 255)
            prediction = deep_face_model.predict_verbose(mask_applied)
            
            if prediction['dominant_gender'] == label:
                print("predict correct: ", prediction['dominant_gender'])
                predicted_male += 1
            else:
                print("predict wrong: ", prediction['dominant_gender'])
            
            # label = ""
            # if ethnicity == 'East Asian':
            #     label = 'asian'
            # if prediction['dominant_race'] == ethnicity.lower() or prediction['dominant_race'] == label:
            #     print("predict correct: ", prediction['dominant_race'])
            #     predicted_male += 1
            # else:
            #     print("predict wrong: ", prediction['dominant_race'])
    label = "Woman"             
    for i in female_images:
        if females < 1000:
            prepared_image = cv2.imread(image_dir + df.iloc[i]['file'])
            
            females += 1
            # prepared_image = np.multiply(prepared_image[0], 255).astype(np.uint8)[0]
            mask_applied = apply_accessory(prepared_image, color_acc, mask)
            
            # cv2.imshow("image", mask_applied)
            # cv2.waitKey(1)
            
            mask_applied = np.expand_dims(mask_applied, axis=0)
            mask_applied = mask_applied.astype(np.float32)
            mask_applied = np.divide(mask_applied, 255)
            prediction = deep_face_model.predict_verbose(mask_applied)
            if prediction['dominant_gender'] == label:
                print("predict correct: ", prediction['dominant_gender'])
                predicted_female += 1
            else:
                print("predict wrong: ", prediction['dominant_gender'])

    return males, females, predicted_male, predicted_female
                    
f = open("D:/Github/capstone-project-team-31/experiment/gender_results.txt", "w")
for ethnicity in ethnicities:
    for colour in colours:
        for accessory in accessories:
            
            color_acc, mask = prepare_accessory(colour, f"D:/Github/capstone-project-team-31/experiment/assets/{accessory}.png", accessory)
            males, females, predicted_male, predicted_female = 0, 0, 0, 0
            males, females, predicted_male, predicted_female = fairface_eval(males, females, predicted_male, predicted_female)
            
            print(f"Accuracy on predicting male ({males} images, ethnicity: {ethnicity}, accessory type: {accessory}, accessory colour: {colour}): {predicted_male / males}")
            f.write(f"Accuracy on predicting male ({males} images, ethnicity: {ethnicity}, accessory type: {accessory}, accessory colour: {colour}): {predicted_male / males}\n")
            print(f"Accuracy on predicting female ({females} images, ethnicity: {ethnicity}, accessory type: {accessory}, accessory colour: {colour}): {predicted_female / females}")
            f.write(f"Accuracy on predicting female ({females} images, ethnicity: {ethnicity}, accessory type: {accessory}, accessory colour: {colour}): {predicted_female / females}\n")
            
            # print(f"Accuracy on predicting male ({males} images, ethnicity: {ethnicity}): {predicted_male / males}")
            # f.write(f"Accuracy on predicting male ({males} images, ethnicity: {ethnicity}): {predicted_male / males}\n")
            # print(f"Accuracy on predicting female ({females} images, ethnicity: {ethnicity}): {predicted_female / females}")
            # f.write(f"Accuracy on predicting female ({females} images, ethnicity: {ethnicity}): {predicted_female / females}\n")
f.close()