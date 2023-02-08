import cv2
import os
from os import path
import mediapipe as mp
import numpy as np
from data_vis.display import NumpyArrayEncoder, mediapipe_detection, extract_keypoints_no_face_mirror, extract_keypoints_no_face_raw
import shutil
import json
from drivers.video import IntelCamera, StandardCamera
import math

class CustomImageDataset():
    # class CustomImageDataset(Dataset):
    def __init__(self, sequence_length, DIR_PATH, RESOLUTION_X,RESOLUTION_Y = True):        
        self.sequence_length = sequence_length
        self.DATA_PATH_TRAIN =path.join(DIR_PATH+'/dataset/MP_Data/Train/')
        self.DATA_PATH_VALID =path.join(DIR_PATH+'/dataset/MP_Data/Valid/')
        self.DATA_PATH_TEST =path.join(DIR_PATH+'/dataset/MP_Data/Test/')

        self.RESOLUTION_X = RESOLUTION_X
        self.RESOLUTION_Y = RESOLUTION_Y

        if os.path.exists(path.join(DIR_PATH, 'config.json')):
            with open(path.join(DIR_PATH, "config.json"), "r") as f:
                config = json.load(f)
                if (
                    "type" in config["camera"]
                    and "width" in config["camera"]
                    and "height" in config["camera"]
                ):
                    if config["camera"]["type"] == "standard":
                        self.cap = StandardCamera(
                            config["camera"]["width"], config["camera"]["height"], config["camera"]["number"]
                        ) if "number" in config["camera"] else StandardCamera(
                            config["camera"]["width"], config["camera"]["height"]
                        )
                    elif config["camera"]["type"] == "intel":

                        self.cap = IntelCamera(
                            config["camera"]["width"], config["camera"]["height"], config["camera"]["number"]
                        ) if "number" in config["camera"] else IntelCamera(
                            config["camera"]["width"], config["camera"]["height"]
                        )
                else:
                    self.cap = StandardCamera(1920, 1080, 0)

                if ("nb_sequences" in config["model_params"] 
                    and "adapt_for_mirror" in config["model_params"]
                    and "erase_dataset" in config["model_params"]
                    and "actions" in config["model_params"]
                    ):
                    self.actions = np.array(config["model_params"]["actions"])
                    self.nb_sequences = config["model_params"]["nb_sequences"]
                    self.adapt_for_mirror = config["model_params"]["adapt_for_mirror"]
                    self.erase_dataset = config["model_params"]["erase_dataset"]

        
        for action in self.actions:
            if(self.erase_dataset):
                if(os.path.exists(self.DATA_PATH_TRAIN + action)):
                    shutil.rmtree(os.path.join(
                        self.DATA_PATH_TRAIN, action))
                if(os.path.exists(self.DATA_PATH_VALID + action)):
                    shutil.rmtree(os.path.join(
                        self.DATA_PATH_VALID, action), ignore_errors=False, onerror=None)
                if(os.path.exists(self.DATA_PATH_TEST + action)):
                    shutil.rmtree(os.path.join(
                        self.DATA_PATH_TEST, action), ignore_errors=False, onerror=None)

    # def __len__(self): return len(self.actions)*len(self.nb_sequences)

    def __getitem__(self):
        mp_holistic = mp.solutions.holistic  # Holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
        model_complexity=0,
        refine_face_landmarks = True) as holistic:
            for action in self.actions:
                # On compte le nombre de séquences déjà enregistrées pour chaque action
                if(os.path.exists(os.path.join(self.DATA_PATH_TRAIN, action))):
                    local_train_seq_nb = len(os.listdir(os.path.join(self.DATA_PATH_TRAIN, action)))
                else:
                    os.makedirs(os.path.join(
                        self.DATA_PATH_TRAIN, action))
                    local_train_seq_nb = 0

                if(os.path.exists(os.path.join(self.DATA_PATH_VALID, action))):
                    local_valid_seq_nb = len(os.listdir(os.path.join(self.DATA_PATH_VALID, action)))
                else:
                    os.makedirs(os.path.join(
                        self.DATA_PATH_VALID, action))
                    local_valid_seq_nb = 0

                if(os.path.exists(os.path.join(self.DATA_PATH_TEST, action))):
                    local_test_seq_nb = len(os.listdir(os.path.join(self.DATA_PATH_TEST, action)))
                else:
                    os.makedirs(os.path.join(
                        self.DATA_PATH_TEST, action))
                    local_test_seq_nb = 0

                # Si on a déjà assez de séquences, on passe à l'action suivante
                if (local_train_seq_nb + local_valid_seq_nb + local_test_seq_nb) >= self.nb_sequences:
                    continue

                # On enregistre les séquences
                for seq_ind in range(local_train_seq_nb, int(math.ceil(self.nb_sequences*80/100))):
                    self.collect_sequence(action, seq_ind, self.DATA_PATH_TRAIN, holistic)

                for seq_ind in range(local_valid_seq_nb, int(math.ceil(self.nb_sequences*10/100))):
                    self.collect_sequence(action, seq_ind, self.DATA_PATH_VALID, holistic)

                for seq_ind in range(local_test_seq_nb, int(math.ceil(self.nb_sequences*10/100))):
                    self.collect_sequence(action, seq_ind, self.DATA_PATH_TEST, holistic)

                cv2.destroyAllWindows()

    def collect_sequence(self, action, seq_ind, PATH, holistic):
        try:
            # On créee les dossiers pour les séquences
            os.makedirs(os.path.join(
                PATH, action, str(seq_ind)))
        except Exception as e:
            raise(e)
        
        # On affiche un message pour indiquer que la collection commence
        # image = np.zeros((640,480,3), np.uint8)
        image = np.zeros((1920, 1080,3), np.uint8)

        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(image, 'Collecting frames for {}'.format(action), (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, 'Video Number {}'.format(seq_ind), (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)
        cv2.waitKey(2000)

        for frame_num in range(-4, self.sequence_length):
            if frame_num<0:
                continue
            
            # Set mediapipe model
            frame, _ = self.cap.next_frame()
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)

            cv2.putText(image, '{}'.format(action), (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, 'Video Number {}'.format(seq_ind), (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            
            if (self.adapt_for_mirror):
                keypoints = extract_keypoints_no_face_mirror(results, image.shape[1], image.shape[0])
            else :
                keypoints = extract_keypoints_no_face_raw(results)

            npy_path = os.path.join(
                PATH, action, str(seq_ind), str(frame_num))

            with open(npy_path+".json", "w") as outfile:
                outfile.write(json.dumps(keypoints, cls=NumpyArrayEncoder))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break   