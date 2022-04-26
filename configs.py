import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def m1(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n
  
  
def m2(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    
    n.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n
  
def m3(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    n.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n

def m4(n):
    n.fc = nn.Linear(n.fc.in_features, 18)
    return n

def m5(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5), bias=False)
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n
  
def m6(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(2, 2), padding=(5, 5), bias=False)
    
    n.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n

def m7(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), bias=False)
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n
  
def m8(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(13, 13), stride=(2, 2), padding=(6, 6), bias=False)
    
    n.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[0].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[1].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv1 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer1[2].conv2 = nn.Conv2d(64, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer2[0].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[1].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[2].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv1 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer2[3].conv2 = nn.Conv2d(128, 128, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer3[0].conv1 = nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer3[0].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[1].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[2].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[3].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[4].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv1 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer3[5].conv2 = nn.Conv2d(256, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    n.layer4[0].conv1 = nn.Conv2d(256, 512, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    n.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[1].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv1 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    n.layer4[2].conv2 = nn.Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
    
    
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n
    
def m9(n):
    n.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
    n.fc = nn.Linear(n.fc.in_features, 18)
    
    return n
    
configs = [
    # {"layer_function" : m1,
    # "save_path": '../freq_bias/models/Fairface_Multi_layer0_3_layerall_3_epoch_13_trial2.pt',
    # },
    # {"layer_function" : m2,
    # "save_path": '../freq_bias/models/UTKFace_Multi_layer0_3_layerall_7_epoch_13_trial1.pt',
    # },
    # {"layer_function" : m3,
    # "save_path": '../freq_bias/models/Fairface_Multi_layer0_7_layerall_7_epoch_13_trial2.pt',
    # },
    # {"layer_function" : m4,
    # "save_path": '../freq_bias/models/Fairface_Multi_layer0_7_layerall_3_epoch_13_trial2.pt',
    # },
    # {"layer_function" : m5,
    # "save_path": '../freq_bias/models/Fairface_Multi_layer0_11_layerall_3_epoch_13_trial2.pt',
    # },
    # {"layer_function" : m6,
    # "save_path": '../freq_bias/models/UTKFace_Multi_layer0_11_layerall_7_epoch_13_trial1.pt',
    # },
    # {"layer_function" : m7,
    # "save_path": '../freq_bias/models/Fairface_Multi_layer0_13_layerall_3_epoch_13_trial2.pt',
    # },
    # {"layer_function" : m8,
    # "save_path": '../freq_bias/models/UTKFace_Multi_layer0_13_layerall_7_epoch_13_trial1.pt',
    # },
    {"layer_function" : m9,
    "save_path": '../freq_bias/models/Fairface_Multi_layer0_5_layerall_3_epoch_13_trial1.pt',
    },
    {"layer_function" : m9,
    "save_path": '../freq_bias/models/Fairface_Multi_layer0_5_layerall_3_epoch_13_trial2.pt',
    },
]
