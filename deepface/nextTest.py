from deepface.DeepFace import *

import matplotlib.pyplot as plt
import tensorflow as tf


actions = ['age', 'gender', 'race', 'emotion']

def getObjects(img_path,
    actions=("emotion", "age", "gender", "race"),
    enforce_detection=True,
    detector_backend="ssd", # set it to [opencv, ssd, dlib, mtcnn, retinaface, mediapipe, yolov8 face, yunet] for face alignment
    align=True,
    silent=False,
):
    # validate actions
    if isinstance(actions, str):
        actions = (actions,)

    # check if actions is not an iterable or empty.
    if not hasattr(actions, "__getitem__") or not actions:
        raise ValueError("`actions` must be a list of strings.")

    actions = list(actions)

    # For each action, check if it is valid
    for action in actions:
        if action not in ("emotion", "age", "gender", "race"):
            raise ValueError(
                f"Invalid action passed ({repr(action)})). "
                "Valid actions are `emotion`, `age`, `gender`, `race`."
            )
    # ---------------------------------
    # build models
    models = {}
    if "emotion" in actions:
        models["emotion"] = build_model("Emotion")

    if "age" in actions:
        models["age"] = build_model("Age")

    if "gender" in actions:
        models["gender"] = build_model("Gender")

    if "race" in actions:
        models["race"] = build_model("Race")
    # ---------------------------------
    img_objs = functions.extract_faces(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    
    return (img_objs, models)
    

def runObjects(img_objs, models,
               actions=("emotion", "age", "gender", "race"),
               silent=False):
    resp_objects = []
    for img_content, img_region, _ in img_objs:

        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            obj = {}
            # facial attribute analysis
            pbar = tqdm(range(0, len(actions)), desc="Finding actions", disable=silent)
            for index in pbar:
                action = actions[index]
                pbar.set_description(f"Action: {action}")

                if action == "emotion":
                    img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.resize(img_gray, (48, 48))
                    img_gray = np.expand_dims(img_gray, axis=0)

                    emotion_predictions = models["emotion"].predict(img_gray, verbose=0)[0, :]

                    sum_of_predictions = emotion_predictions.sum()

                    obj["emotion"] = {}

                    for i, emotion_label in enumerate(Emotion.labels):
                        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                        obj["emotion"][emotion_label] = emotion_prediction

                    obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

                elif action == "age":
                    age_predictions = models["age"].predict(img_content, verbose=0)[0, :]
                    apparent_age = Age.findApparentAge(age_predictions)
                    # int cast is for exception - object of type 'float32' is not JSON serializable
                    obj["age"] = int(apparent_age)

                elif action == "gender":
                    gender_predictions = models["gender"].predict(img_content, verbose=0)[0, :]
                    obj["gender"] = {}
                    for i, gender_label in enumerate(Gender.labels):
                        gender_prediction = 100 * gender_predictions[i]
                        obj["gender"][gender_label] = gender_prediction

                    obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

                elif action == "race":
                    race_predictions = models["race"].predict(img_content, verbose=0)[0, :]
                    sum_of_predictions = race_predictions.sum()

                    obj["race"] = {}
                    for i, race_label in enumerate(Race.labels):
                        race_prediction = 100 * race_predictions[i] / sum_of_predictions
                        obj["race"][race_label] = race_prediction

                    obj["dominant_race"] = Race.labels[np.argmax(race_predictions)]

                # -----------------------------
                # mention facial areas
                obj["region"] = img_region

            resp_objects.append(obj)

    return resp_objects

import random
from copy import deepcopy
def small_noise(input_array, delta = 1):
    output = deepcopy(input_array)
    for i in range(len(input_array)):
        o = input_array[i] + random.uniform(-delta, delta)
        output[i] = o
    return output
    

def repeat_noise(img_objs, models, target, actions, targetStrength = 0.5):
    oldOutput = runObjects(img_objs,models,actions)[0]
    oldConfidence = oldOutput['gender']['Woman']
    testObjects = deepcopy(img_objs)
    oldImage = deepcopy(testObjects[0][0][0])
    newImage = deepcopy(oldImage)
    while oldConfidence <= targetStrength:
        
        curI = random.randint(0,len(oldImage)-1)
        curJ = random.randint(0,len(oldImage[curI])-1)
        added_noise = small_noise(oldImage[curI][curJ])
        newImage = deepcopy(oldImage)
        newImage[curI][curJ] = added_noise
        testObjects[0][0][0] = newImage
        newOutput = runObjects(testObjects, models, actions)[0]
        newConfidence = newOutput['gender']['Woman']
        print(oldConfidence, newConfidence)
        if newConfidence > oldConfidence:
            oldConfidence = newConfidence
            oldImage = deepcopy(newImage)
            
    return oldImage
