import numpy as np
import cv2
import numpy as np
import mediapipe as mp
import json
from os import path
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def extract_keypoints_no_face_mirror(results, width, height):
    
    face_lm_ind = [10, 152, 234, 454]
    pose = np.array([[res.x * width, res.y * height] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*2)
    face = np.array([[res.x * width, res.y * height] for res in [results.face_landmarks.landmark[i] for i in face_lm_ind]]).flatten(
    ) if results.face_landmarks else np.zeros(4*2)
    lh = np.array([[res.x * width, res.y * height] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*2)
    rh = np.array([[res.x * width, res.y * height] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*2)
    return np.concatenate([face, pose, lh, rh])

def extract_keypoints_no_face_raw(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return {"body_pose": pose, "left_hand": lh, "right_hand": rh}

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def draw_styled_landmarks(image, results):
        mp_holistic = mp.solutions.holistic
        mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(
                                    color=(80, 22, 10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(
                                    color=(80, 44, 121), thickness=2, circle_radius=2)
                                )
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(
                                    color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(
                                    color=(121, 44, 250), thickness=2, circle_radius=2)
                                )
        # Draw right hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(
                                    color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(
                                    color=(121, 44, 250), thickness=2, circle_radius=2)
                                )

def mediapipe_detection(frame, mp_model):
    image = frame.copy()
    window = 0.7

    min_width, max_width = int((0.5 - window / 2) * frame.shape[1]), int(
        (0.5 + window / 2) * frame.shape[1]
    )
    if path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
                if ("flip" in config["camera"]): 
                    if config["camera"]["flip"] == True:
                        image = cv2.flip(image, 1)
                
    image = image[:, min_width:max_width]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

    return image, results

def prob_viz(sign, probability, input_frame, colors, action):
    
    output_frame = input_frame.copy()
    indCol = 0
    cv2.rectangle(output_frame, (0, 60),
                    (int(probability*len(sign)*1.5), 90),
                    colors[indCol], -1)
    cv2.putText(output_frame, sign, (0, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(output_frame, (0, 560),
                    (100, 590),
                    colors[1], -1)
    cv2.putText(output_frame, action, (0, 585),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame