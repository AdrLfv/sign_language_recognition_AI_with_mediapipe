import random

class Data_augmentation():

    def __init__(self, data_frame,):

        # on récupère chaque frame de chaque sequence de chaque action, on effectue les modifications sur celle-ci,
        # on sauvegarde les données de cette nouvelle frame dans une nouvelle séquence dans la même action,
        # on effectue sur la frame suivante les mêmes modifications et on sauvegarde dans la nouvelle séquence
        # lorsque l'on passe à la séquence suivante on change les paramètres de modification
        
        width = 640
        height = 440
        x_shift = random.uniform(-0.5, 0.5)*width
        y_shift = random.uniform(-0.5, 0.5)*height
        scale = random.uniform(0.5, 1.5)
        self.new_data_frame = []
        x_max = max(data_frame[x] for x in range(0,len(data_frame),2))
        y_max = max(data_frame[y] for y in range(1,len(data_frame),2))
        x_min = min(data_frame[x] for x in range(0,len(data_frame),2))
        y_min = min(data_frame[y] for y in range(1,len(data_frame),2))
        #print("data frame : ", np.array(data_frame).shape)
        for ind, _ in enumerate(data_frame[0:]):
            if (ind % 2 == 0):
                self.new_data_frame.append( (data_frame[ind]-(x_max+x_min)/2) * scale + x_shift + (x_max+x_min)/2 if data_frame[ind] != 0 else 0)
                self.new_data_frame.append( (data_frame[ind+1]-(y_max+y_min)/2) * scale + y_shift + (y_max+y_min)/2 if data_frame[ind+1] != 0 else 0)

                # self.new_data_frame.append(data_frame[ind] + x_shift)
                # self.new_data_frame.append(data_frame[ind+1] + y_shift)

    def __getitem__(self):
        return self.new_data_frame