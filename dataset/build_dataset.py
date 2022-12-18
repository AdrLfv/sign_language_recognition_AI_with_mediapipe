import cv2
import os
import mediapipe as mp
import numpy as np
from ..data_vis.display import NumpyArrayEncoder, mediapipe_detection, extract_keypoints_no_face_mirror, extract_keypoints_no_face_raw
import shutil
import json
import time


class CustomImageDataset():
    # class CustomImageDataset(Dataset):
    def __init__(self, actionsToAdd, nb_sequences, sequence_length, DATA_PATH, RESOLUTION_X,RESOLUTION_Y, SOURCE, adapt_for_mirror = True):
        self.actionsToAdd = actionsToAdd
        self.nb_sequences = nb_sequences
        self.sequence_length = sequence_length
        self.DATA_PATH_TRAIN =DATA_PATH+"/Train/"
        self.DATA_PATH_VALID =DATA_PATH+"/Valid/"
        self.DATA_PATH_TEST =DATA_PATH+"/Test/"
        self.RESOLUTION_X = RESOLUTION_X
        self.RESOLUTION_Y = RESOLUTION_Y
        self.cap = SOURCE
        self.adapt_for_mirror = adapt_for_mirror
        # self.cap = IntelCamera()
        print('dataset init')

        for action in self.actionsToAdd:
            if(os.path.exists(self.DATA_PATH_TRAIN + action)):
                shutil.rmtree(os.path.join(
                    self.DATA_PATH_TRAIN, action))
            if(os.path.exists(self.DATA_PATH_VALID + action)):
                shutil.rmtree(os.path.join(
                    self.DATA_PATH_VALID, action), ignore_errors=False, onerror=None)
            if(os.path.exists(self.DATA_PATH_TEST + action)):
                shutil.rmtree(os.path.join(
                    self.DATA_PATH_TEST, action), ignore_errors=False, onerror=None)

            os.makedirs(os.path.join(
                self.DATA_PATH_TRAIN, action))
            os.makedirs(os.path.join(
                self.DATA_PATH_VALID, action))
            os.makedirs(os.path.join(
                self.DATA_PATH_TEST, action))

    def __len__(self): return len(self.actionsToAdd)*len(self.nb_sequences)

    def __getitem__(self):
        for action in self.actionsToAdd:
            for sequence in range(self.nb_sequences):
                try:
                    if(sequence<self.nb_sequences*80/100):
                        os.makedirs(os.path.join(
                            self.DATA_PATH_TRAIN, action, str(sequence)))
                    elif(self.nb_sequences*80/100 <= sequence and sequence < self.nb_sequences*90/100):
                        os.makedirs(os.path.join(
                            self.DATA_PATH_VALID, action, str(int(sequence-self.nb_sequences*80/100))))
                    else:
                        os.makedirs(os.path.join(
                            self.DATA_PATH_TEST, action, str(int(sequence-self.nb_sequences*90/100))))
                except Exception as e:
                    print(e)
                    pass
        
        # Set mediapipe model
        mp_holistic = mp.solutions.holistic  # Holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
        model_complexity=0,
        refine_face_landmarks = True) as holistic:
        
            for action in self.actionsToAdd:
                for sequence in range(self.nb_sequences):
                    
                    image = np.zeros((1920,1080,3), np.uint8)
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {}'.format(action), (15, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Video Number {}'.format(sequence), (15, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)

                    for frame_num in range(-4, self.sequence_length):
                        frame, _ = self.cap.next_frame()
                        frame = cv2.flip(frame, 1)
                        image, results = mediapipe_detection(frame, holistic)

                        cv2.putText(image, 'Collecting frames for {}'.format(action), (15, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, 'Video Number {}'.format(sequence), (15, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                        if frame_num<0:
                            continue
                        if (self.adapt_for_mirror):
                            keypoints = extract_keypoints_no_face_mirror(results)
                        else :
                            keypoints = extract_keypoints_no_face_raw(results)

                        if(sequence<self.nb_sequences*80/100):
                            npy_path = os.path.join(
                            self.DATA_PATH_TRAIN, action, str(sequence), str(frame_num))
                        elif(self.nb_sequences*80/100 <= sequence and sequence < self.nb_sequences*90/100):
                            npy_path = os.path.join(
                            self.DATA_PATH_VALID, action, str(int(sequence-self.nb_sequences*80/100)), str(frame_num))
                        else:
                            npy_path = os.path.join(
                            self.DATA_PATH_TEST, action, str(int(sequence-self.nb_sequences*90/100)), str(frame_num))
                        with open(npy_path+".json", "w") as outfile:
                            outfile.write(json.dumps(keypoints, cls=NumpyArrayEncoder))

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break   

            cv2.destroyAllWindows()