from torchvision import models
import torch.nn as nn
from datahandler import trainset



model = models.resnext50_32x4d(pretrained=True)

inputs = model.fc.in_features
outputs = len(trainset.classes)

clf = nn.Sequential(
              nn.Dropout(0.30), 
              nn.Linear(inputs, outputs)
                  )

model.fc = clf