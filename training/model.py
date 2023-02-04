from torch import nn  
import torch.nn.functional as F
import torch
torch.cuda.empty_cache()

class myLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, layer_size, output_size, device, bidirectional=True, ):
        super(myLSTM, self).__init__()

        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, output_size
        self.bidirectional = bidirectional
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size,
                            batch_first=True, bidirectional=bidirectional).to(device)

        if bidirectional:  # we'll have 2 more layers
            self.layer = nn.Linear(hidden_size*2, output_size).to(device)
        else:
            self.layer = nn.Linear(hidden_size, output_size).to(device)

    def forward(self, input):
        # input = input.to(self.device)
        
        # Set initial states
        if self.bidirectional:
            hidden_state = torch.zeros(
                self.layer_size*2, input.size(0), self.hidden_size, device=self.device)
            cell_state = torch.zeros(
                self.layer_size*2, input.size(0), self.hidden_size, device=self.device)
        else:
            hidden_state = torch.zeros(
                self.layer_size, input.size(0), self.hidden_size, device=self.device)
            cell_state = torch.zeros(
                self.layer_size, input.size(0), self.hidden_size, device=self.device)

        # LSTM:
        output, (last_hidden_state, last_cell_state) = self.lstm(input)
        # output, (last_hidden_state, last_cell_state) = self.lstm(input, (hidden_state, cell_state))
        
        # Reshape
        output = output[:, -1, :]

        # FNN:
        output = self.layer(output)

        return output
        # return output, (last_hidden_state, last_cell_state)