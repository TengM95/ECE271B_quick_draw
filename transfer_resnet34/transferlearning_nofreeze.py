#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# In[2]:


import torch
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F 


image_info = pd.read_csv("/datasets/ChestXray-NIHCC/Data_Entry_2017.csv")
labels = image_info["Finding Labels"]
classes = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 
                3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 
                7: "Pneumothorax", 8: "Consolidation", 9: "Edema", 
                10: "Emphysema", 11: "Fibrosis", 
                12: "Pleural_Thickening", 13: "Hernia",14:"No Finding" }
class_label  = {v: k for k, v in classes.items()}
label_stats = []
for l in labels: 
    l_list = l.split('|')
    
    for z in l_list:
        label_stats.append(class_label[z])

data = label_stats

# fixed bin size
bins = np.arange(0, 16, 1) # fixed bin size

plt.xlim([min(data)-1, max(data)+1])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('label_stats')
plt.xlabel('labels'"")
plt.ylabel('count')

plt.show()

counts, b, bars = plt.hist(data,bins=bins)
print(len(counts))
# print((counts))
minus_counts = 112120 - counts
# print(minus_counts)

weight_bce = minus_counts/counts
# print(weight_bce)
ratio = 1.0*counts/sum(counts)

weight_bce = weight_bce[0:-1]
# print(weight_bce.shape)
weight_temp = np.ones((len(weight_bce)))

weight_bce = torch.from_numpy(np.array(np.log10(weight_bce)*18)).type(torch.cuda.FloatTensor)


# In[3]:


weight_bce


# In[4]:


import torchvision.models as models
model = models.resnet34(pretrained=True)
model.fc = nn.Sequential(nn.Linear(512, 14))


# In[5]:


#ct = 0
#for child in model.children():
#    ct += 1
#    if ct < 9:
#        for param in child.parameters():
#            param.requires_grad = False


# In[6]:


class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):
        
        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)
        
        weight = Variable(self.weight)
           
        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


# In[7]:


from baseline_cnn import *


# Setup: initialize the hyperparameters/variables
# Setup: initialize the hyperparameters/variables
num_epochs = 10           # Number of full passes through the dataset
batch_size = 128          # Number of samples in each minibatch
learning_rate = 0.001  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing


#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here

transform = transforms.Compose([
        #transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

model = model
model = model.to(computing_device)
    
# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)

criterion = FocalLoss2d(weight=weight_bce).to(computing_device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[8]:


print("Model on CUDA?", next(model.parameters()).is_cuda)


# In[9]:


import torch.nn.functional as F


# In[10]:


import numpy as np
# label = np.asarray([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
# output = np.asarray([[0, 0.9, 0], [0, 0.9, 0], [0.9, 0, 0.6]])
def prediction(labels, outputs):
    threshold = 0.5
    num_cor = 0
    num_tp = 0
    num_fp = 0
    num_fn = 0
    num_total = labels.shape[0]*labels.shape[1]
    test_result = np.zeros((labels.shape))
    
    temp_position = np.where(outputs >= threshold)
    
    test_result[temp_position] = 1
        
    temp_position0 = np.array(np.where(test_result == 1.0)).T.tolist()
    temp_position1 = np.array(np.where(labels == 1)).T.tolist()
          
    temp_position2 = np.array(np.where(test_result == 0.0)).T.tolist()
    temp_position3 = np.array(np.where(labels == 0)).T.tolist()
    
    temp_position4 = np.array(np.where(labels == test_result)).T.tolist()
    
    
    num_cor = len(temp_position4)
    for element in temp_position0:
        if element in temp_position1:
            num_tp += 1
    
    for element in temp_position0:
        if element in temp_position3:
            num_fp += 1   
    
    for element in temp_position2:
        if element in temp_position1:
            num_fn += 1      
  
    accuracy = num_cor/num_total
    if num_fp + num_tp == 0:
        precision = 0
    else:
        precision = num_tp/(num_fp + num_tp)
    if num_tp + num_fn == 0:
        recall = 0
    else:
        recall = num_tp/(num_tp + num_fn)
    bcr = (precision + recall)/2
    return accuracy, precision, recall, bcr
def validate(val_loader,model,optimizer):
  
    start = time.time()
    sum_loss = 0.0
    list_sum_loss = []
    num = 0
    for mb_count, (val_images, val_labels) in enumerate(val_loader, 0):
        model.eval()
        with torch.no_grad():  
            optimizer.zero_grad()      
            val_images = torch.squeeze(torch.stack([val_images,val_images,val_images], dim=1, out=None))
            val_images, val_labels = val_images.to(computing_device), val_labels.to(computing_device)
            outputs = model(val_images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs,val_labels)
            sum_loss += loss
    print("validation time = ", time.time()-start)    
    return 1.0*sum_loss/mb_count  


# In[15]:


def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)


# In[16]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()
    total_loss = []
    avg_minibatch_loss = []
    total_vali_loss = []
    tolerence = 3
    i = 0 
    for epoch in range(num_epochs):
        N = 50
        M = 50
        N_minibatch_loss = 0.0    
        early_stop = 0
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            scheduler.step()
            # Iterate over data.
            for minibatch_count, (inputs, labels) in enumerate(train_loader, 0):
                inputs = torch.squeeze(torch.stack([inputs,inputs,inputs], dim=1, out=None))
                inputs = inputs.to(computing_device)
                labels = labels.to(computing_device)
                
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    N_minibatch_loss += loss
                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                
                # statistics
                if minibatch_count % N == 0 and minibatch_count!=0:    

                    # Print the loss averaged over the last N mini-batches    
                    N_minibatch_loss /= N
                    print('Epoch %d, average minibatch %d loss: %.3f' %
                        (epoch + 1, minibatch_count, N_minibatch_loss))

                    # Add the averaged loss over N minibatches and reset the counter
                    avg_minibatch_loss.append(N_minibatch_loss)
                    N_minibatch_loss = 0.0

                    output_np = outputs.cpu().detach().numpy()
                    label_np = labels.cpu().detach().numpy()

                    accuracy, precision, recall, bcr = prediction(label_np, output_np)
                    print('accuracy, precision, recall', accuracy, precision, recall)
                if minibatch_count % M == 0 and minibatch_count!=0: 
                    #model = torch.load('./checkpoint')
                    save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                },
                                filename='./checkpoint/'+'%d_model_epochnew%d.pth' % (epoch , minibatch_count))
                    v_loss = validate(val_loader,model,optimizer).item()
                    print(v_loss)
                    total_vali_loss.append(v_loss)
                    
                    if total_vali_loss[i] > total_vali_loss[i-1] and i != 0:
                        early_stop += 1
                        if early_stop == tolerence:

                            avg_minibatch_loss_1 = np.array(avg_minibatch_loss)
                            np.save('avg_minibatch_loss_new', avg_minibatch_loss_1)

                            total_vali_loss_1 = np.array(total_vali_loss)
                            np.save('total_vali_loss_new', total_vali_loss_1)                    

                            print('early stop here')
                            break
                    else:
                        early_stop = 0
                    i = i + 1
            print("Finished", epoch + 1, "epochs of training")
    print("Training complete after", epoch, "epochs")
    
    avg_minibatch_loss = np.array(avg_minibatch_loss)
    np.save('avg_minibatch_loss_new', avg_minibatch_loss)

    total_vali_loss = np.array(total_vali_loss)
    np.save('total_vali_loss_new', total_vali_loss)  
    print("total_vali_loss")
    print(total_vali_loss)
    print("avg_minibatch_loss")
    print(avg_minibatch_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s '.format(
        time_elapsed // 60, time_elapsed % 60))


# In[17]:


model_trained = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




