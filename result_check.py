# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 08:56:20 2023

@author: Anish Hilary
"""

import torch

result_dir = r'C:\Users\Anish Hilary\RESNET\normal_models_cifar\results\20_09-23_22_80\resnet_34/latest_model.pth'

result_dict = torch.load(result_dir)

print(f"The total epochs : {result_dict['epoch']}")
print(f"Best accuracy : {max(result_dict['valid_epoch_accuracy'])}")
#print(f"Best accuracy : {result_dict['best_accuracy']}")
print(f"Parameters : {result_dict['learnable_params']}")