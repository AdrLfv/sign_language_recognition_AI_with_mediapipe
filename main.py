import numpy as np
import os
from os import path
import json
import onnxruntime as ort

from data_vis.tuto import Tuto
from dataset.build_dataset import CustomImageDataset

from models_conversion.pthToOnnx import export_to_onnx

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm  # For nice progress bar!
from training.model import myLSTM
from training.preprocess import Preprocess
import shutil

def test_loop(loader, model, device, writer, epoch, name):
    num_correct = 0
    num_samples = 0
    model.eval().to(device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    model.train()
    writer.add_scalar(f"{name} Accuracy", num_correct/num_samples, epoch)
    return num_correct/ num_samples

def main():
    DIR_PATH = path.dirname(path.realpath(__file__))
    DATA_PATH = path.join(DIR_PATH, 'dataset/MP_Data')

    RESOLUTION_Y = int(1920)  # Screen resolution in pixel
    RESOLUTION_X = int(1080)

    if os.path.exists(path.join(DIR_PATH, "config.json")):
        with open(path.join(DIR_PATH, "config.json"), "r") as f:
            config = json.load(f)
            if (
                "make_dataset" in config["model_params"]
                and "make_train" in config["model_params"]
                and "convert_files" in config["model_params"]
                and "make_data_augmentation" in config["model_params"]
                and "make_tuto" in config["model_params"]
                and "sequence_length" in config["model_params"]
                and "actions" in config["model_params"]
                and "nb_epochs" in config["model_params"]
                and "erase_runs" in config["model_params"]
            ):
                # The user will have to make the dataset for the actions in "actionsToAdd"
                make_dataset = config["model_params"]["make_dataset"]
                # The program will launch the train
                make_train = config["model_params"]["make_train"]
                # The program will convert the weights to pth
                convert_files = config["model_params"]["convert_files"]
                # The program will make data augmentation
                make_data_augmentation = config["model_params"]["make_data_augmentation"]
                # The program will make a data visualisation of the recording
                make_tuto = config["model_params"]["make_tuto"]
                # The program will automatically calibrate the coordinates for the mirror

                sequence_length = config["model_params"]["sequence_length"]
                actions = np.array(config["model_params"]["actions"])
                # "empty", "nothing", 'hello', 'thanks', 'iloveyou', "what's up", "hey", "my", "name", "nice","to meet you", "ok", "left", "right"
                nb_epochs = config["model_params"]["nb_epochs"]
                erase_runs = config["model_params"]["erase_runs"]

    if (erase_runs):
        for file in os.listdir(path.join(DIR_PATH, 'runs')):
            shutil.rmtree(path.join(DIR_PATH, 'runs', file))

    file_ind = 1
    while(os.path.exists(path.join(DIR_PATH, 'runs/slr_lstm0_'+str(file_ind))) == True): file_ind += 1
    writer = SummaryWriter(path.join(DIR_PATH, 'runs/slr_lstm0_'+str(file_ind)))

    output_size = len(actions)
    hidden_size = 128
    num_layers = 2

    if (make_dataset):
        CustomImageDataset(sequence_length, DIR_PATH, RESOLUTION_X, RESOLUTION_Y).__getitem__()

    #myTestOnnx = TestOnnx()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if(make_train):
        learning_rate = 1e-4  # how much to update models parameters at each batch/epoch
        batch_size = 16  # number of data samples propagated through the network before the parameters are updated
        # NUM_WORKERS = 4
        NUM_WORKERS = 6
        DECAY = 1e-4

        # on crée des instances de preprocess en leur donnant le chemin d'accès ainsi que le nombre de séquences dans chaque dossier
        # en fonction de si leur type de preprocess est train, valid, test.

        train_preprocess = Preprocess(
            actions, DATA_PATH+"/Train", sequence_length, make_data_augmentation, device)
        valid_preprocess = Preprocess(
            actions, DATA_PATH+"/Valid", sequence_length, False, device)
        test_preprocess = Preprocess(
            actions, DATA_PATH+"/Test", sequence_length, False, device)

        input_size = train_preprocess.get_data_length()

        train_loader = DataLoader(train_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=True)

        test_loader = DataLoader(test_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=True)

        valid_loader = DataLoader(valid_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                                pin_memory=True)

        # Initialize network
        try:
            # model path: path.join(DIR_PATH, '/outputs/slr_' + str(output_size) + '.pth')
            model = myLSTM(input_size, hidden_size,
                num_layers, output_size, device)
            if os.path.exists(path.join(DIR_PATH, 'outputs/slr_' + str(output_size) + '.pth')):
                model.load_state_dict(torch.load(path.join(DIR_PATH, 'outputs/slr_' + str(output_size) + '.pth')))
                print("Model loaded")

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=DECAY)

            frames, _ = next(iter(train_loader))

            writer.add_graph(model, frames.to(device))
            writer.close()

            #* Train Network
            running_loss = 0.0
            running_correct = 0.0

            for epoch in range(nb_epochs):
                print(f"Epoch {epoch+1}\n-------------------------------")
                with tqdm(train_loader, desc="Train") as pbar:
                    running_loss = 0.0
                    model = model.train()

                    # for data, targets in enumerate(tqdm(train_loader)):
                    for i, (frame, targets) in enumerate(pbar):
                        frame, targets = frame.cuda(), targets.cuda()
                        #frame, targets = frame.cuda().float(), targets.cuda().float()
                        optimizer.zero_grad()
                        outputs = model(frame)

                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() #* à ajouter ou non / len(train_loader)

                        _, predicted = torch.max(outputs.data, 1)
                        running_correct += (predicted == targets).sum().item()

                    running_train_acc = float(running_correct) / float(len(train_loader.dataset))
                        # if(i+1) % 10 == 0:
                    writer.add_scalar('Training loss', loss, epoch)
                    # writer.add_scalar('Training accuracy', running_train_acc, epoch)

                    running_loss = 0.0
                    running_correct = 0.0

                print(
                    f"Accuracy on training set: {test_loop(train_loader, model, device, writer, epoch, 'Training')*100:.2f}")
                print(
                    f"Accuracy on test set: {test_loop(test_loader, model, device, writer, epoch, 'Testing')*100:.2f}")
            print("Done!")

            # print(
            #     f"Accuracy on valid set: {test_loop(valid_loader, model, device)*100:.2f}")

            # torch.save(model.state_dict(), path.join(DIR_PATH, 'outputs/slr_'+str(output_size)+'.pth'))

            # export_to_onnx(input_size, hidden_size, num_layers, output_size, device, DIR_PATH)

        except Exception as e:
            raise e
            # print(e)

    else:
        if(convert_files):
            try:
                ort.InferenceSession(path.join(DIR_PATH, '/outputs/slr_' + str(output_size) + '.onnx'), providers=[
                    'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
                print("Found valid onnx model")
            except Exception as e:
                print("Onnx model not found")
                print(e)
                try:
                    print("Converting pth to onnx")
                    export_to_onnx(output_size, hidden_size, num_layers, output_size, device, DIR_PATH)
                except Exception as e:
                    raise(e)
                    print("Unable to convert to onnx")

                raise(e)


    if make_tuto:
        myTuto = Tuto(actions, RESOLUTION_X, RESOLUTION_Y, DATA_PATH)
        for action in actions:
            if (action != "nothing" and action != "empty"):
                myTuto.launch_tuto(action)

if __name__ =='__main__':
    main()
