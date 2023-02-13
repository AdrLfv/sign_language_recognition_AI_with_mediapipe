import json
import os
import math


DATA_FOLDER = 'dataset/MP_Data'
SUBFOLDERS = ['Train', 'Valid', 'Test']

PREVIOUS_SEQ_NB = 30
NEXT_SEQ_NB = 90
TRAIN_RATIO = int(math.ceil((NEXT_SEQ_NB - PREVIOUS_SEQ_NB)*80/100))
TEST_RATIO = int(math.ceil((NEXT_SEQ_NB - PREVIOUS_SEQ_NB)*10/100))

def fix_data():
    for subfolder in SUBFOLDERS:
        if subfolder == 'Train': ratio = TRAIN_RATIO
        else: ratio = TEST_RATIO
        actions = os.listdir(os.path.join(DATA_FOLDER, subfolder))
        for action in actions:
            folders = os.listdir(os.path.join(DATA_FOLDER, subfolder, action))
            for folder in folders:
                folder_nb = int(folder.replace('.json', ''))
                os.rename(os.path.join(DATA_FOLDER, subfolder, action, folder), os.path.join(DATA_FOLDER, subfolder, action, str(folder_nb+ratio)))
                    
    print("Done!")

if __name__ == '__main__':
    fix_data()


import json
import os
import math
import shutil


PREVIOUS_DATA_FOLDER = 'dataset/MP_Data_16_act_50_seq'
NEXT_DATA_FOLDER = 'dataset/MP_Data'
SUBFOLDERS = ['Train', 'Valid', 'Test']

PREVIOUS_SEQ_NB = 30
NEXT_SEQ_NB = 60
TRAIN_RATIO = int(math.ceil((NEXT_SEQ_NB - PREVIOUS_SEQ_NB)*80/100))
TEST_RATIO = int(math.ceil((NEXT_SEQ_NB - PREVIOUS_SEQ_NB)*10/100))

def fix_data():
    for subfolder in SUBFOLDERS:
        if subfolder == 'Train': ratio = TRAIN_RATIO
        else: ratio = TEST_RATIO
        actions = os.listdir(os.path.join(PREVIOUS_DATA_FOLDER, subfolder))
        for action in actions:
            folders = os.listdir(os.path.join(PREVIOUS_DATA_FOLDER, subfolder, action))
            for folder in folders:
                folder_nb = int(folder.replace('.json', ''))
                os.rename(os.path.join(PREVIOUS_DATA_FOLDER, subfolder, action, folder), os.path.join(NEXT_DATA_FOLDER, subfolder, action, str(folder_nb+ratio)))
                files = os.listdir(os.path.join(PREVIOUS_DATA_FOLDER, subfolder, action, str(folder_nb+ratio)))
                for file in files:
                    json_previous_file = os.path.join(PREVIOUS_DATA_FOLDER, subfolder, action, folder, file)
                    json_next_file = os.path.join(NEXT_DATA_FOLDER, subfolder, action, folder, file)
                    if os.path.isfile(json_previous_file):
                        shutil.copy(json_previous_file, json_next_file)

    print("Done!")

if __name__ == '__main__':
    fix_data()
