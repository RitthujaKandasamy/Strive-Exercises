import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
#from torchsummary import summary




# constructe the neural network archiecture
def NeuralNetwork(input_layer, hidden_layer):

	
       mlpmodel = nn.Sequential(
        nn.Linear(input_layer, hidden_layer[0]),  
        nn.Linear(),
        nn.Linear(hidden_layer[0], hidden_layer[1]),  
        nn.ReLU(),
        nn.Linear(hidden_layer[1], hidden_layer[2]),  
        nn.ReLU(),
        nn.Linear(hidden_layer[2], hidden_layer[3]), 
		nn.Linear(),
		nn.Linear(hidden_layer[3], 1)

    )
       return mlpmodel