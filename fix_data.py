import json
import os

DATA_FOLDER = 'dataset/MP_Data'
SUBFOLDERS = ['Train', 'Valid', 'Test']

FROM = [640, 480]
TO = [448, 480]
RATIO = [TO[0] / FROM[0], TO[1] / FROM[1]]

def fix_data():
    for subfolder in SUBFOLDERS:
        subsubfolders = os.listdir(os.path.join(DATA_FOLDER, subfolder))
        for fd in subsubfolders:
            actions = os.listdir(os.path.join(DATA_FOLDER, subfolder, fd))
            for action in actions:
                files = os.listdir(os.path.join(DATA_FOLDER, subfolder, fd, action))
                for file in files:
                    json_file = os.path.join(DATA_FOLDER, subfolder, fd, action, file)
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    data = [data[i] * RATIO[0] if i % 2 == 0 else data[i] * RATIO[1] for i in range(len(data))]
                    with open(json_file, 'w') as f:
                        json.dump(data, f)
    print("Done!")

if __name__ == '__main__':
    fix_data()
