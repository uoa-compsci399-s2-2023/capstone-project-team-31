import pandas as pd
from image_helper_functions import *
from deepface_models import *

class ResultsCSV:
    
    def __init__(self, images_dir, num_images):
        
        self.images_dir = images_dir
        self.num_images = num_images
        
        
    def run(self):
        
        images = prepare_processed_images(self.images_dir, self.num_images)
        
        cols = ['True Gender', 'Classified Gender', 'Gender Accuracy', 'True Ethnicity', 'Classified Ethnicity', 'Ethnicity Accuracy', 'True Mood', 'Classified Mood', 'Mood Accuracy', 'True Age', 'Classified Age', 'Age Accuracy']
        
        df = pd.DataFrame(columns=cols)
        
        gender_model = attributeModel("gender")
        ethnicity_model = attributeModel("ethnicity")
        mood_model = attributeModel("emotion")
        age_model = attributeModel("age")
        
        for image in images:
            
            img_data = []
            
            img_contents = cleanup_dims(image[0])
            
            true_gender = image[2]
            gender_prediction = gender_model.predict_verbose(img_contents)
            img_data += [true_gender, gender_prediction['dominant_gender'], None]
            
            print(gender_prediction)
            
            true_ethnicity = image[1]
            ethnicity_prediction = ethnicity_model.predict_verbose(img_contents)
            img_data += [true_ethnicity, ethnicity_prediction['dominant_race'], None]
            
            print(ethnicity_prediction)
            
            true_mood = image[4]
            mood_prediction = mood_model.predict_verbose(img_contents)
            img_data += [true_mood, mood_prediction['dominant_emotion'], None]
            print(mood_prediction)
            
            true_age = image[3]
            age_prediction = age_model.predict_verbose(img_contents)
            img_data += [true_age, age_prediction['apparentAge'], abs(int(true_age) - int(age_prediction['apparentAge']))]
            print(age_prediction)
            
            df.loc[len(df)] = img_data
                
        
        df.to_csv('results/stats.csv')
        
        
def cleanup_dims(image):
    
    ## cleaning up dimension issues:
    if len(np.shape(image)) == 3:
        image = np.expand_dims(image, axis=0)
        
    return image