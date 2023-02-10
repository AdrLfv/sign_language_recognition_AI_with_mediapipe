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
