import numpy as np
import os
import json
import onnxruntime as ort
import torch

from .dataset.build_dataset import CustomImageDataset

from .training.LSTM import myLSTM
from .training.preprocess import Preprocess
from .training.test import Test

from .models_conversion.pthToOnnx import export_to_onnx
from .models_conversion.test_onnx import TestOnnx

from .data_vis.tuto import Tuto

# Gives easier dataset managment by creating mini batches etc.
from torch.utils.data import DataLoader
from tqdm import tqdm  # For nice progress bar!

from .drivers.video import IntelCamera, StandardCamera 


def launch_LSTM(output_size, train, weights_type, make_data_augmentation):
    # Set device cuda for GPU if it's available otherwise run on the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    learning_rate = 0.001  # how much to update models parameters at each batch/epoch
    batch_size = 32  # number of data samples propagated through the network before the parameters are updated
    NUM_WORKERS = 4
    num_epochs = 100  # number times to iterate over the dataset
    DECAY = 1e-4
    hidden_size = 128  # number of features in the hidden state h
    num_layers = 2

    # on crée des instances de preprocess en leur donnant le chemin d'accès ainsi que le nombre de séquences dans chaque dossier
    # en fonction de si leur type de preprocess est train, valid, test.
    train_preprocess = Preprocess(
        actions, DATA_PATH_TRAIN, nb_sequences_train, sequence_length, make_data_augmentation)
    valid_preprocess = Preprocess(
        actions, DATA_PATH_VALID, nb_sequences_valid, sequence_length, False)
    test_preprocess = Preprocess(
        actions, DATA_PATH_TEST, nb_sequences_test, sequence_length, False)

    input_size = train_preprocess.get_data_length()

    train_loader = DataLoader(train_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)

    test_loader = DataLoader(test_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                             pin_memory=True)

    valid_loader = DataLoader(valid_preprocess, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    # Initialize network
    

    if(train):  # On verifie si on souhaite reentrainer le modele
        
        # export_to_onnx(output_size)
        if(weights_type == "onnx"):
            try:
                ort.InferenceSession("../outputs/slr_"+str(output_size)+".onnx", providers=[
                    'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
                print("Found valid onnx model")
            except Exception as e:
                print("Onnx model not found")
                print(e)
                try:
                    export_to_onnx(output_size)
                    print("Converted pth to onnx")
                except:
                    print("Unable to convert to onnx")
                    weights_type = "pth"
            model = None

        if(weights_type == "pth"):
            model = myLSTM(input_size,  hidden_size,
                        num_layers, output_size).to(device)
            try:
                model.load_state_dict(torch.load(
                    "../outputs/slr_"+str(output_size)+".pth"))
                print("Found valid pth model")
            except Exception as e:
                print("Pth model not found")
                
                model = train_launch(model, output_size, learning_rate, DECAY,
                                    num_epochs, train_loader, test_loader, valid_loader)
            export_to_onnx(output_size)

    return model  # ,logits


def train_loop(train_loader, model, criterion, optimizer):
    with tqdm(train_loader, desc="Train") as pbar:
        total_loss = 0.0
        model = model.train()
        # for data, targets in enumerate(tqdm(train_loader)):
        for frame, targets in pbar:
            frame, targets = frame.cuda(), targets.cuda()
            #frame, targets = frame.cuda().float(), targets.cuda().float()
            optimizer.zero_grad()
            # Get to correct shape
            scores = model(frame)
            # print(targets.shape)
            # print(scores.shape)
            loss = criterion(scores, targets)
            # backward
            loss.backward()
            # gradient descent or adam step
            optimizer.step()
            total_loss += loss.item() / len(train_loader)
            pbar.set_postfix(loss=total_loss)
    # Check accuracy on training & test to see how good our model


def test_loop(loader, model, criterion):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            y = y.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            # x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)
            
    model.train()
    return num_correct/ num_samples


def train_launch(model, output_size, learning_rate, DECAY, num_epochs, train_loader, test_loader, valid_loader):
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=DECAY)
    # Train Network

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(train_loader, model, criterion, optimizer)

        print(
            f"Accuracy on training set: {test_loop(train_loader, model, criterion)*100:.2f}")
        print(
            f"Accuracy on test set: {test_loop(test_loader, model, criterion)*100:.2f}")
    print("Done!")

    print(
        f"Accuracy on valid set: {test_loop(valid_loader, model, criterion)*100:.2f}")

    torch.save(model.state_dict(), './outputs/slr_'+str(output_size)+'.pth')
    
    return model


# on crée des dossiers dans lequels stocker les positions des points que l'on va enregistrer
DATA_PATH_TRAIN = os.path.join('slr_mirror/dataset/MP_Data/Train')
DATA_PATH_VALID = os.path.join('slr_mirror/dataset/MP_Data/Valid')
DATA_PATH_TEST = os.path.join('slr_mirror/dataset/MP_Data/Test')

RESOLUTION_Y = int(1920)  # Screen resolution in pixel
RESOLUTION_X = int(1080)

# Thirty videos worth of data
nb_sequences = 1
nb_sequences_train = int(nb_sequences*80/100)
nb_sequences_valid = int(nb_sequences*10/100)
nb_sequences_test = int(nb_sequences*10/100)

# Videos are going to be 30 frames in length
sequence_length = 30

make_train = True
make_dataset = True
make_data_augmentation = False
make_tuto = False
weights_type = "onnx"  # "pth"

if(make_dataset):
    make_train = True

# dataset making : (ajouter des actions dans le actionsToAdd pour créer leur dataset)
actionsToAdd = np.array(["empty", "nothing"])  #

# Actions that we try to detect
actions = np.array(["nothing", "empty"])
# , "nothing" 'hello', 'thanks', 'iloveyou', "what's up", "hey", "my", "name", "nice","to meet you"

if os.path.exists("config.json"):
    with open("config.json", "r") as f:
        config = json.load(f)
        if (
            "type" in config["camera"]
            and "width" in config["camera"]
            and "height" in config["camera"]
        ):
            if config["camera"]["type"] == "standard":
                SOURCE = StandardCamera(
                    config["camera"]["width"], config["camera"]["height"], config["camera"]["number"]
                ) if "number" in config["camera"] else StandardCamera(
                    config["camera"]["width"], config["camera"]["height"]
                )
            elif config["camera"]["type"] == "intel":

                SOURCE = IntelCamera(
                    config["camera"]["width"], config["camera"]["height"], config["camera"]["number"]
                ) if "number" in config["camera"] else IntelCamera(
                    config["camera"]["width"], config["camera"]["height"]
                )
        else:
            SOURCE = StandardCamera(1920, 1080, 0)
else:
    SOURCE = StandardCamera(1920, 1080, 0)

if (make_dataset):
    CustomImageDataset(actionsToAdd, nb_sequences, sequence_length, DATA_PATH_TRAIN,
    DATA_PATH_VALID, DATA_PATH_TEST, RESOLUTION_X, RESOLUTION_Y, SOURCE).__getitem__()

#myTestOnnx = TestOnnx()

model = launch_LSTM(len(actions), make_train, weights_type, make_data_augmentation)

if make_tuto:
    if(weights_type == "pth"):
        myTest = Test(model, len(actions))
    if(weights_type == "onnx"):
        myTest = TestOnnx(len(actions))
    myTuto = Tuto(actions, RESOLUTION_X, RESOLUTION_Y)
    for action in actions:
        if (action != "nothing" and action != "empty"):
            myTuto.launch_tuto(action)
            myTest.launch_test(actions, action, SOURCE, RESOLUTION_X, RESOLUTION_Y)