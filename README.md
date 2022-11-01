# Sign Language Training module with Artificial Intelligence recognition

## Description

**This project was done as part of my studies at the De Vinci Innovation Center as an assignment for the Artificial Intelligence course. The objective was to implement a module for learning sign language on an augmented reality platform.** 

**At the time of the launching of the program, the user sees appearing a character drawn with features, which makes him a demonstration of a word in sign language. He must then repeat this movement by having a return on the camera of the device. He can see in real time the points detected on his body. When he has validated the movement, a second tutorial appears about another movement and he must do it again, and so on.**

You can find my work on my GitHub : https://github.com/AdrLfv

## Requirements

- A nvidia GPU with drivers installed
- OpenCV and Mediapipe ( !pip install opencv-python mediapipe )

## Before launching the program

Before running the program, you may want to change some settings depending on whether this is the first time you run the program or not.

You have to precise if you want to launch the train (automatic if no weight is found in the directory), to make the dataset (automatic if no dataset is found in *MP_Data*) and to use the data augmentation, from *lines 113 to 115* in the main. The variables are make_train, make_dataset, and make_data_augmentation.

    make_train =  True
    make_dataset = False
    make_data_augmentation = True

The creation of the dataset is optional, you must enter before launching the program which movements you want to (re)record in the array "actionsToAdd" (*line 120* in the main). 

    actionsToAdd = ["empty"]

The actions recognized by the IA are regrouped in the array "actions" (*line 123* in the main). You may want adjust some.

    actions = np.array(["nothing","empty", "hello", "thanks", "iloveyou", "what's up",  "my", "name","nice","to meet you"])


All the actions recognized by the IA are not presented by the tutorial. Indeed, some actions such as "nothing" (when the user does nothing) and "empty" (when nobody is detected in front of the camera) are not desired in the launch tutorial. You should want to add them *line 136* in the main: 
    
    if (action != "nothing" and action != "empty")

## Details about the program

* The main (`main.py`)

As mentioned before, the main initializes the different parameters most common to the different classes and calls one by one the different functions constituting the program.

* The dataset (`dataset.py`)


If a movement is included in the known actions but is not found in the files, the creation of the dataset of this movement is launched automatically.

It is stored in *MP data* created, and separated in three folders : *train*, *valid*, *test* containing the actions (names of the movements). The recorded sequences are automatically separated between the three folders *train* (80\% of them), *test* (10\%) and *valid* (10\%) (those percentages can be modified *line 106, 107 and 108*).

    nb_sequences_train = int(nb_sequences*80/100)
    nb_sequences_valid = int(nb_sequences*10/100)
    nb_sequences_test = int(nb_sequences*10/100)
    
Each sequence is 30 frames each, including coordinates, that is to say 258 data per frame after having removed the numerous coordinates of the face (435 points) which were not useful.
The coordinates are retrieved from `Mediapipe` during the creation of the dataset.
The recording is done from the webcam of the device used, here `cv2` with VideoCapture(0). 

*  The tutorial (`tuto.py`)

The tutorial module will retrieve the coordinates of a video by action, stored in the dataset. 

It sorts the coordinates of each point of each part of the body, it finds the different links between each point and the posters, frame by frame. 
For this, arrays of all the sorted coordinates for each body part (Face, Body, Hand) are used, as well as arrays of all the links of the body.

For each point detected in the dataset, we display the links it has with the points it is linked to in the table.
We also display the action to which the tutorial is associated, above the drawn character.

* The pre-process (`preprocess.py`)
    
The preprocess phase retrieves all the data from the dataset and places them in tensors to provide them to the model. 
    
After creating the preprocess instances (according to their type: train, test, valid) we pass them to the dataloader. These pass an index to them (concerning a sequence among all the sequences attributed to the type of the preprocess). The program calculates which action is concerned by the desired sequence and recovers all the coordinates of the frames of this sequence.
    
Currently the preprocess does not get the coordinates of the face, this allows to have much faster train loops because all these points are not necessary.

If you may want to add the train taking into consideration the facial points you have to remove the call to `remove_face` *line 58* in the preprocess :

    res = remove_face(res)
    
* The data augmentation (`data_augmentation.py`)
    
The data augmentation is called during the pre-processing phase, just after retrieving the data of the concerned sequence. It is given the data as a parameter, it reviews the data and will apply a random horizontal and vertical shift and scale to it. This allows to artificially increase the number of positions in which the user can stand.
    
* The model (`LSTM.py`)

The model is launched from the main, all the parameters are defined there, as well as the preprocess instances and the calls to the DataLoader. 
    
If the program is unable to access an already pre-trained and stored model, the model training is launched. 
This one is made of a bidirectional LSTM (two layers), a bidirectional linear and a classical gradient descent.
    
    
* The user test (`test.py`)

During the test, the coordinates of the user are recovered and placed in tensors, we obtain a tensor containing the data of a video of 30 frames. 
    
The model is called by passing the data to it and a table of probabilities concerning each action is retrieved as a result. 
These probabilities are then displayed by a rectangle containing the most probable action, as well as a coloring of the rectangle more or less important according to the probability.

## Ressources:

- https://google.github.io/mediapipe/solutions/holistic.html
- https://google.github.io/mediapipe/solutions/hands.html

## Support

You can contact me on : adrien.lefevre@edu.devinci.fr

## Execute

In the root folder : python3 -m slr_mirror