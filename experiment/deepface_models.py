from deepface.DeepFace import *

import matplotlib.pyplot as plt
loss_object = tf.keras.losses.CategoricalCrossentropy()

class attributeModel: 
    
    def __init__(self, action):
        self.action = action
        if "emotion"  == action:          
            self.model = build_model("Emotion")
            self.labels = Emotion.labels

        elif "age" == action:
            self.model = build_model("Age")
            self.labels = Age.labels
        elif "gender"  == action:
            self.model = build_model("Gender")
            self.labels = Gender.labels
        elif "race" == action:
            self.model = build_model("Race")
            self.labels = Race.labels
        else:
            raise ValueError(
                f"Invalid action passed ({repr(action)})). "
                "Valid actions are `emotion`, `age`, `gender`, `race`."
            )
            
    
            
    def prediction_to_labels(self, prediction):
        output = {}
        if self.action == "emotion":

            sum_of_predictions = prediction.sum()
            output["emotion"] = {}

            for i, emotion_label in enumerate(Emotion.labels):
                emotion_prediction = 100 * prediction[i] / sum_of_predictions
                output["emotion"][emotion_label] = emotion_prediction

            output["dominant_emotion"] = Emotion.labels[np.argmax(prediction)]
            output["classfied"] = output["dominant_emotion"]
            
        elif self.action == "age":
            apparent_age = Age.findApparentAge(prediction)
            # int cast is for exception - object of type 'float32' is not JSON serializable
            output["age"] = int(apparent_age)
            output["classified"] = output["age"]
        elif self.action == "gender":
            output["gender"] = {}
            for i, gender_label in enumerate(Gender.labels):
                gender_prediction = 100 * prediction[i]
                output["gender"][gender_label] = gender_prediction

            output["dominant_gender"] = Gender.labels[np.argmax(prediction)]
            output["classified"] = output["dominant_gender"]
            
        elif self.action == "race":
            sum_of_predictions = prediction.sum()

            output["race"] = {}
            for i, race_label in enumerate(Race.labels):
                race_prediction = 100 * prediction[i] / sum_of_predictions
                output["race"][race_label] = race_prediction

            output["dominant_race"] = Race.labels[np.argmax(prediction)]
            output["classified"] = output["dominant_race"]
            
        output["confidence"] = prediction.max() / prediction.sum()
        output["max_index"] = np.argmax(prediction)
        return output
    
    def raw_predict(self,img_content):
        predictions = self.model.predict(img_content, verbose=0)[0, :]
        return predictions

    def generateLabelFromIndex(self, index):
        rawList = len(self.labels) * [0.]
        outputList = np.array(rawList)
        outputList[index] = 1
        return outputList
    
    def generateLabelFromText(self, text):
        index = self.labels.index(text)
        return self.generateLabelFromIndex(index)
    
    def image_resize(self, img):
        if self.action == "emotion":
            img_gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            img_gray = np.expand_dims(img_gray, axis=0)

            return img_gray
        else:
            return img
    
    def predict(self, img_content):
        '''
        Given a (1,224,224,3) image predicts using given model
        
        Outputs: List of unlabeled predictions
        '''
        resized_img = self.image_resize(img_content)
        prediction = self.raw_predict(resized_img)
        return prediction
        
    def predict_verbose(self,img_content):          
        '''
        Given a (1,224,224,3) image predicts using given model
        
        Outputs: Dictionary of labeled predictions
        '''
        prediction = self.predict(img_content)
        output = self.prediction_to_labels(prediction)
        return output
    
    def find_gradient(self, input_image_np, input_label_raw):
        '''
        Given a (1,224,224,3) tensor predicts using given model
        Requires input_image to be converted to tensor - can be done using tf.convert_to_tensor
        
        Outputs: Gradient respective to input_label
        
        '''

        input_label = np.expand_dims(input_label_raw, axis = 0)
        input_image = tf.convert_to_tensor(input_image_np)
        
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = self.model(input_image)
            loss = loss_object(input_label, prediction)
        gradient = tape.gradient(loss, input_image)
        
        return gradient
    
    def find_resized_gradient(self, input_image, input_label):
        '''
        Same as find_gradient but resized if you have emotion
        '''
        
        if(self.action != "emotion"):
            gradient = self.find_gradient(input_image, input_label)
            return gradient
        gray = self.image_resize(input_image)
        gradient = self.find_gradient(gray, input_label)
        np_gradient = gradient.numpy()
        resized = cv2.resize(np_gradient[0], (224, 224))
        gray = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        return gray
        
    
    def create_adversarial_pattern(self, input_image, input_label):
        gradient = self.find_gradient(input_image, input_label)
        signed_grad = tf.sign(gradient)
        return signed_grad

#################################################################
# Short demo

# import experiment.image_helper_functions as img_helper
# e = attributeModel("emotion")
# image_list = img_helper.prepare_processed_images("Faces.db",3)
# currentImage = image_list[0]

# print("Before Prediction", e.predict_verbose(currentImage[0]))
# actual_label = e.generateLabelFromText(currentImage[4])
# tens = tf.convert_to_tensor(currentImage[0])
# gradient = e.find_resized_gradient(currentImage[0], actual_label)

# attackIm = currentImage[0] + gradient * 0.1
# print("After Prediction", e.predict_verbose(attackIm))

