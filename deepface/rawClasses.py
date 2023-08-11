from deepface.DeepFace import *

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
            
            
            
    def raw_unprocessed_predict(self, img_content):
        if(self.action == "emotion"):
            img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            img_gray = np.expand_dims(img_gray, axis=0)
            return self.raw_predict(img_gray)
        else:
            return self.raw_predict(img_content)
    
    def raw_predict(self,img_content):
        predictions = self.model.predict(img_content, verbose=0)[0, :]
        return predictions
    
    def predict(self,img_content):      
        output = {}
        if self.action == "emotion":
            img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            img_gray = np.expand_dims(img_gray, axis=0)

            emotion_predictions = self.raw_predict(img_gray)

            sum_of_predictions = emotion_predictions.sum()
            output["emotion"] = {}

            for i, emotion_label in enumerate(Emotion.labels):
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                output["emotion"][emotion_label] = emotion_prediction

            output["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

        elif self.action == "age":
            age_predictions = self.raw_predict(img_content)
            apparent_age = Age.findApparentAge(age_predictions)
            # int cast is for exception - object of type 'float32' is not JSON serializable
            output["age"] = int(apparent_age)

        elif self.action == "gender":
            gender_predictions = self.raw_predict(img_content)
            output["gender"] = {}
            for i, gender_label in enumerate(Gender.labels):
                gender_prediction = 100 * gender_predictions[i]
                output["gender"][gender_label] = gender_prediction

            output["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

        elif self.action == "race":
            race_predictions = self.raw_predict(img_content)
            sum_of_predictions = race_predictions.sum()

            output["race"] = {}
            for i, race_label in enumerate(Race.labels):
                race_prediction = 100 * race_predictions[i] / sum_of_predictions
                output["race"][race_label] = race_prediction

            output["dominant_race"] = Race.labels[np.argmax(race_predictions)]
        return output
    
def getImageObjects(img_path,
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
):
    img_objs = functions.extract_faces(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    
    return img_objs
    
def getImageContents(img_path,
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
):
    img_objs = getImageObjects(img_path, 
                               enforce_detection = enforce_detection,
                               align = align, 
                               detector_backend = detector_backend)
    contents = []
    for (content, region, _) in img_objs:
        contents.append(content)
        
    return contents
    
