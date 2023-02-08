import time
import cv2
import numpy as np
import os
import json


BODY_LINKS = [
    [[0, 1], [0, 4], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 8]],
    [[9, 10]],
    [
        [11, 12],
        [11, 13],
        [11, 23],
        [12, 14],
        [12, 24],
        [13, 15],
        [14, 16],
        [15, 17],  # 7
        [15, 19],  # 8
        [15, 21],  # 9
        [16, 18],  # 10
        [16, 20],  # 11
        [16, 22],  # 12
        [17, 19],  # 13
        [18, 20],  # 14
        [23, 24],
        [23, 25],
        [24, 26],
        [25, 27],
        [26, 28],
        [27, 29],
        [27, 31],
        [28, 30],
        [28, 32],
    ],
]

HAND_LINKS = [
    [
        [0, 1],
        [0, 5],
        [0, 9],
        [0, 13],
        [0, 17],
        [5, 9],
        [9, 13],
        [13, 17]
    ],
    [
        [1, 2],
        [2, 3],
        [3, 4]
    ],
    [
        [5, 6],
        [6, 7],
        [7, 8]
    ],
    [
        [9, 10],
        [10, 11],
        [11, 12]
    ],
    [
        [13, 14],
        [14, 15],
        [15, 16]
    ],
    [
        [17, 18],
        [18, 19],
        [19, 20]
    ]
]


class Tuto:
    """ Use Body parametters to draw the body on a provided image """

    def __init__(self,actions, RESOLUTION_X,RESOLUTION_Y, DATA_PATH):
        #1680 6920
        self.body_junctions = BODY_LINKS
        self.hand_junctions = HAND_LINKS
        # self.face_junctions = FACE_LINKS

        self.color = (255, 255, 255)
        self.thickness = 2

        self.show_head = False
        self.show_wrist = True

        self.actions = actions

        self.RESOLUTION_X = RESOLUTION_X
        self.RESOLUTION_Y = RESOLUTION_Y

        self.DATA_PATH = DATA_PATH

    def draw(self, image, data, action):
        """ Draws the body on an opencv image """
        body_pose = data["body"]
        # left_hand_pose = data["left_hand"]
        # right_hand_pose = data["right_hand"]
        # face_pose = data["face"]

        # image = cv2.line(image, [200, 200], [200, 600], self.color, self.thickness)

        if len(body_pose) >= 0:
            for i, parts in enumerate(self.body_junctions):
                if i > 1:
                    for j, pair in enumerate(parts):
                        if j < 7 or j > 14:
                            image = cv2.line(
                                image,
                                body_pose[pair[0]],
                                body_pose[pair[1]],
                                self.color,
                                self.thickness,
                            )

        # if len(left_hand_pose) >= 0:
        #     for i, parts in enumerate(self.hand_junctions):
        #         for pair in parts:
        #             image = cv2.line(
        #                 image,
        #                 left_hand_pose[pair[0]],
        #                 left_hand_pose[pair[1]],
        #                 self.color,
        #                 self.thickness,
        #             )

        # if len(right_hand_pose) >= 0:
        #     for i, parts in enumerate(self.hand_junctions):
        #         for pair in parts:
        #             image = cv2.line(
        #                 image,
        #                 right_hand_pose[pair[0]],
        #                 right_hand_pose[pair[1]],
        #                 self.color,
        #                 self.thickness,
        #             )
        cv2.putText(image, 'Action : {}'.format(action), (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
        # if len(self.face_pose) >= 0:
        #     # print(len(self.face_pose))
        #     for i, parts in enumerate(self.face_junctions):
        #         if(i >= 0):
        #             for i in range(len(parts) - 1):
        #                 image = cv2.line(
        #                     image,
        #                     self.face_pose[parts[i]],
        #                     self.face_pose[parts[i+1]],
        #                     self.color,
        #                     self.thickness,
        #                 )

        return image

    def launch_tuto(self, action):
        # Actions that we try to detect
        sequence_length = 30
        sequence = []

        image = np.zeros((self.RESOLUTION_Y, self.RESOLUTION_X, 3), dtype=np.uint8)

        # Recuperation des donnees du dataset
        sequence_idx = 0
        for frame_num in range(sequence_length):
            dataPath = open(os.path.join(self.DATA_PATH,"Train", action, str(sequence_idx), "{}.json".format(frame_num)))
            res = json.load(dataPath)
            sequence.append(res)

        # frame = sequence[0]
        # data = {}
        # pose_landmarks = [[int(frame[i]), int(frame[i+1])]
        #     for i, _ in enumerate(frame[0:33*2]) if i % 2 == 0]

        # left_hand_landmarks = [[int(frame[i]), int(frame[i+1])]
        #     for i, _ in enumerate(frame[33*2: 33*2+21*2]) if i % 2 == 0]
                                
        # right_hand_landmarks = [[int(frame[i]), int(
        #     frame[i+1])] for i, _ in enumerate(frame[33*2+21*2:]) if i % 2 == 0]
        
        # data["body"] = pose_landmarks
        # data["left_hand"] = left_hand_landmarks
        # data["right_hand"] = right_hand_landmarks

        # image = self.draw(image, data)
        # cv2.imshow("My window", image)
        # cv2.waitKey(5000) #millisecondes entre chaque frame

        for frame in sequence:
            data = {}
            pose_landmarks = [[int(frame[i]), int(frame[i+1])]
                for i, _ in enumerate(frame[0:33*2]) if i % 2 == 0]

            # face_landmarks = [[0,0] for i in range(468)]

            left_hand_landmarks = [[int(frame[i]), int(frame[i+1])]
                for i, _ in enumerate(frame[33*2: 33*2+21*2]) if i % 2 == 0]
                                    
            right_hand_landmarks = [[int(frame[i]), int(
                frame[i+1])] for i, _ in enumerate(frame[33*2+21*2:]) if i % 2 == 0]
            
            data["body"] = pose_landmarks
            data["left_hand"] = left_hand_landmarks
            data["right_hand"] = right_hand_landmarks

            image = self.draw(image, data, action)
            cv2.imshow("My window", image)
            key = cv2.waitKey(66) #millisecondes entre chaque frame

            if key == ord('q'):
                break
        
        time.sleep(2)
        cv2.destroyWindow("My window")
