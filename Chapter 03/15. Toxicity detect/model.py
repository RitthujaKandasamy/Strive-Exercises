
import torch.nn as nn
import torch.nn.functional as F





class Network(nn.Module):
    def __init__(self, max_seq_len, emb_dim, hidden1, hidden2, hidden3):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(max_seq_len*emb_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.out = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)
    
    
    def forward(self, inputs):
        layer1 = F.relu(self.fc1(inputs.squeeze(1).float()))
        layer2 = self.dropout(F.relu(self.fc2(layer1)))
        layer3 = self.dropout(F.relu(self.fc3(layer2)))
        layer4 = self.fc4(layer3)
        out = self.out(layer4)

        return out