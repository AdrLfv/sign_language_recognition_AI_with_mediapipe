import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from data_vis.display import draw_styled_landmarks, mediapipe_detection, extract_keypoints_no_face_mirror, prob_viz
import time


class TestOnnx():
    def __init__(self, output_size):

        self.model = ort.InferenceSession("/home/adrlfv/Documents/ESILV/A4/AI/SLR_project_mirror/outputs/slr_"+str(output_size)+".onnx", providers=[
                    'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
        self.timeloop_start = 0
        print("Launching onnx model")

    def get_sign(self, sequence, actions) -> list:
        """
        Get sign from frames
        """
        
        ort_inputs = {'input': np.array(
            [sequence], dtype=np.float32)}
        out = self.model.run(None, ort_inputs)[-1]

        out = np.exp(out) / np.sum(np.exp(out))
        return (actions[np.argmax(out)], float(np.max(out)))

    def launch_test(self, actions, targeted_action, cap, RESOLUTION_X, RESOLUTION_Y):
        """
        actions l'ensemble des actions, action l'action à réaliser, cap l'instance de IntelCamera
        """
        sequence = []
        sentence = []
        threshold = 0.9

        #cap = cv2.VideoCapture(0)
        count_valid = 0

        # Set mediapipe model
        mp_holistic = mp.solutions.holistic  # Holistic model
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
            model_complexity=0,
            refine_face_landmarks = True) as holistic:

            while True:

                timeloop = time.time()-self.timeloop_start
                while(timeloop < 0.09):
                    timeloop = time.time()-self.timeloop_start
                self.timeloop_start = time.time()
                print("TIMELOOP", timeloop)

                

                # RECUPERATION DES COORDONNEES
                # Read feed
                frame, depth = cap.next_frame()
                #frame= cv2.resize(frame,(RESOLUTION_X,RESOLUTION_Y))

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)
                image = cv2.resize(image, (RESOLUTION_Y, RESOLUTION_X))

                # Draw landmarks
                draw_styled_landmarks(image, results)
                #image = cv2.flip(image, 1)
                window = 0.5
                min_width, max_width = int(
                    (0.5-window/2)*RESOLUTION_Y), int((0.5+window/2)*RESOLUTION_Y)

                image = image[:, min_width:max_width]

                # TEST DU MODELE

                # Creation d'une séquence de frames
                keypoints = extract_keypoints_no_face_mirror(results)
                sequence.append(keypoints)

                sequence = sequence[-30:]
                if len(sequence) == 30:
                    sign, probability = self.get_sign(sequence, actions)

                    # 3. Viz logic
                    if probability > threshold:
                        if len(sentence) > 0:
                            if sign != sentence[-1]:
                                sentence.append(sign)
                        else:
                            sentence.append(sign)

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(sign, probability, image,
                                     self.colors, targeted_action)
                    if(sign == targeted_action):
                        count_valid += 1
                    else:
                        count_valid = 0

                    if(count_valid == 10):
                        print("VALIDATED")
                        break
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('My window', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            # cap.release()
