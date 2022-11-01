import cv2
import os
import mediapipe as mp
import numpy as np
from ..data_vis.display import mediapipe_detection, draw_styled_landmarks, extract_keypoints,extract_keypoints_no_face


class CustomImageDataset():
    # class CustomImageDataset(Dataset):
    def __init__(self, actionsToAdd, nb_sequences, sequence_length, DATA_PATH_TRAIN, DATA_PATH_VALID, DATA_PATH_TEST, RESOLUTION_X,RESOLUTION_Y, SOURCE):
        self.actionsToAdd = actionsToAdd
        self.nb_sequences = nb_sequences
        self.sequence_length = sequence_length
        self.DATA_PATH_TRAIN =DATA_PATH_TRAIN
        self.DATA_PATH_VALID =DATA_PATH_VALID
        self.DATA_PATH_TEST =DATA_PATH_TEST
        self.RESOLUTION_X = RESOLUTION_X
        self.RESOLUTION_Y = RESOLUTION_Y
        self.cap = SOURCE
        # self.cap = IntelCamera()
        print('dataset init')

    def __len__(self): return len(self.actionsToAdd)*len(self.nb_sequences)

    def __getitem__(self):
        for action in self.actionsToAdd:
            for sequence in range(self.nb_sequences):
                try:
                    if(sequence<self.nb_sequences*80/100):
                        os.makedirs(os.path.join(
                        self.DATA_PATH_TRAIN, action, str(sequence)))
                        print("creating folder : ", os.path.join(
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

            # Loop through actionsToAdd
            for action in self.actionsToAdd:
                # Loop through sequences aka videos
                for sequence in range(self.nb_sequences):
                    # Loop through video length aka sequence length

                    if(sequence<self.nb_sequences*80/100):
                        DATA_PATH = self.DATA_PATH_TRAIN
                    elif(self.nb_sequences*80/100 <= sequence and sequence < self.nb_sequences*90/100):
                        DATA_PATH = self.DATA_PATH_VALID
                    else:
                        DATA_PATH = self.DATA_PATH_TEST

                    frame,_ = self.cap.next_frame()
                    image, results = mediapipe_detection(frame, holistic)
                    
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {}'.format(action), (15, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Video Number {}'.format(sequence), (15, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                    cv2.waitKey(2000)

                    for frame_num in range(self.sequence_length):
                        
                        cv2.putText(image, 'Collecting frames for {}'.format(action), (15, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(image, 'Video Number {}'.format(sequence), (15, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                        
                        frame, depth = self.cap.next_frame()
                        image, results = mediapipe_detection(frame, holistic)
                        
                        # NEW Export keypoints
                        keypoints = extract_keypoints_no_face(results)
                        #keypoints represente toutes les donnees d'une frame
                        
                        if(sequence<self.nb_sequences*80/100):
                            npy_path = os.path.join(
                            DATA_PATH, action, str(sequence), str(frame_num))
                        elif(self.nb_sequences*80/100 <= sequence and sequence < self.nb_sequences*90/100):
                            npy_path = os.path.join(
                            DATA_PATH, action, str(int(sequence-self.nb_sequences*80/100)), str(frame_num))
                        else:
                            npy_path = os.path.join(
                            DATA_PATH, action, str(int(sequence-self.nb_sequences*90/100)), str(frame_num))
                        
                        print("saving data in ", npy_path)
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break    
     
            cv2.destroyAllWindows()

        
            
