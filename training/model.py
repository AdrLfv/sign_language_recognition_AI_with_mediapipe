from torch import nn  
import torch.nn.functional as F
import torch
torch.cuda.empty_cache()

class myLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, layer_size, output_size, bidirectional=True):
        super(myLSTM, self).__init__()

        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, output_size
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, layer_size,
                            batch_first=True, bidirectional=bidirectional)

        if bidirectional:  # we'll have 2 more layers
            self.layer = nn.Linear(hidden_size*2, output_size)
        else:
            self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, images):
        # print('images shape:', images.shape)
        
        # Set initial states
        if self.bidirectional:
            hidden_state = torch.zeros(
                self.layer_size*2, images.size(0), self.hidden_size)
            cell_state = torch.zeros(
                self.layer_size*2, images.size(0), self.hidden_size)
        else:
            hidden_state = torch.zeros(
                self.layer_size, images.size(0), self.hidden_size)
            cell_state = torch.zeros(
                self.layer_size, images.size(0), self.hidden_size)

        # LSTM:
        #print("images shape",images.shape)
        output, (last_hidden_state, last_cell_state) = self.lstm(images)
        
        # Reshape
        output = output[:, -1, :]

        # FNN:
        output = self.layer(output)

        return output

    

    