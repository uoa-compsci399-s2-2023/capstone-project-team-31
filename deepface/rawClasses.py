from deepface.DeepFace import *
loss_object = tf.keras.losses.CategoricalCrossentropy()

import matplotlib.pyplot as plt

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
    
    def raw_unprocessed_predict(self, img_content):
        if(self.action == "emotion"):
            img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            img_gray = np.expand_dims(img_gray, axis=0)
            self.lastProcessed = img_gray
            return self.raw_predict(img_gray)
        else:
            self.lastProcessed = img_content
            return self.raw_predict(img_content)
    
    def raw_predict(self,img_content):
        predictions = self.model.predict(img_content, verbose=0)[0, :]
        return predictions
    
    def predict(self,img_content):      
        
        prediction = []
        if self.action == "emotion":
            img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray, (48, 48))
            img_gray = np.expand_dims(img_gray, axis=0)

            prediction = self.raw_predict(img_gray)
        else:
            prediction = self.raw_predict(img_content)
        output = self.prediction_to_labels(prediction)
        return output
    
    def create_adversarial_pattern(self, input_image, input_label):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction =self.model(input_image)
            loss = loss_object(input_label, prediction)
        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient)
        return signed_grad

    def getAttackImage(self, input_image, input_label, eps = 0.05):
        gradient = self.create_adversarial_pattern(input_image, input_label)
        adv_x = input_image + eps * gradient
        adv_x = tf.clip_by_value(adv_x, -1,1)
        rawOutput = self.raw_predict(adv_x)
        labeledOutput = self.prediction_to_labels(rawOutput)
        return (adv_x, labeledOutput)

    def lastProcessedIntoAttack(self, actualLabel, eps = 0.05):
        tens = tf.convert_to_tensor(self.lastProcessed)
        oL = np.zeros_like(self.model.predict(self.lastProcessed))
        oL[0][actualLabel] = 1
        return self.getAttackImage(tens, oL, eps = eps)
    
def display_images(image, description):
    plt.figure()
    plt.imshow(image[0]*0.5+0.5)
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,label, confidence*100))
    plt.show()    
    
def sideBySide(imageA, labelA, imageB, labelB):
    fig = plt.figure()
    
    fig.add_subplot(2, 1, 1)
    plt.imshow(imageA[0][:, :, ::-1])
    plt.axis('off')
    plt.title("Classified - {}, Confidence - {}".format(labelA["classified"], labelA["confidence"]))
    fig.add_subplot(2, 1, 2)
    
    plt.imshow(imageB[0][:, :, ::-1])
    plt.axis('off')
    plt.title("Classified - {}, Confidence - {}".format(labelB["classified"], labelB["confidence"]))


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
    
