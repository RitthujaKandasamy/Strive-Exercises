import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
#from torchsummary import summary




# construct the neural network archiecture
def NeuralNetwork(input_size, hidden_sizes, output_size):

	
       mlpmodel = nn.Sequential(OrderedDict([
          ('fc1',   nn.Linear(input_size, hidden_sizes[0])),
          ('linear1', nn.Linear()),
          ('fc2',   nn.Linear(hidden_sizes[0], hidden_sizes[1])),
          ('relu1', nn.ReLU()),
          ('fc3', nn.Linear(hidden_sizes[1], hidden_sizes[2])),
		  ('relu2', nn.ReLU()),
          ('fc4',   nn.Linear(hidden_sizes[2], hidden_sizes[3])),
          ('linear2', nn.Linear()),
          ('output', nn.Linear(hidden_sizes[3], output_size))

		  ]))
    
       return mlpmodel