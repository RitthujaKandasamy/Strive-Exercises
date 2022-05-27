import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
#from torchsummary import summary





def NeuralNetwork(Features, hidden, target):

	# construct a sequential neural network
	mlpModel = nn.Sequential(OrderedDict([
		("hidden_layer", nn.Linear(Features, hidden)),
		#("activation_function", nn.Linear()),
		("output_layer", nn.Linear(hidden, target))
	]))
	
	return mlpModel
       