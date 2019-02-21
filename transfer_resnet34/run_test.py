#!/usr/bin/env python
# coding: utf-8

# In[78]:


import numpy as np
#labels = np.asarray([[0,0,0],[1, 0, 0], [0, 1, 1], [0, 1, 1]])
labels = np.asarray([ [0, 1, 1]])
#outputs = np.asarray([[0,0,0],[0, 0.9, 0], [0, 0.9, 0], [0.9, 0, 0.6]])
outputs = np.asarray([ [0.9, 0, 0.6]])
threshold = 0.5
def prediction_perclass(labels, outputs):

    test_result = np.zeros((labels.shape))
    temp_position = np.where(outputs >= threshold)
    test_result[temp_position] = 1
    
    temp_position0 = np.array(np.where(test_result == 1.0)).T.tolist()
    temp_position1 = np.array(np.where(labels == 1)).T.tolist()
          
    temp_position2 = np.array(np.where(test_result == 0.0)).T.tolist()
    temp_position3 = np.array(np.where(labels == 0)).T.tolist() 
    temp_position4 = np.array(np.where(labels == test_result)).T.tolist()

    class_tp = np.zeros( labels.shape[1])
    class_fp = np.zeros( labels.shape[1])
    class_fn = np.zeros( labels.shape[1])
    class_ac = np.zeros( labels.shape[1])
    
    accuracy_list = np.zeros( labels.shape[1])
    precision_list = np.zeros( labels.shape[1])
    recall_list = np.zeros( labels.shape[1])
    
    for element in temp_position4:
        class_ac[element[1]] += 1
    
    for element in temp_position0:

        if element in temp_position1:
            class_tp[element[1]] += 1
        else:
            class_fp[element[1]] += 1

    for element in temp_position2:
        if element in temp_position1:
            class_fn[element[1]] += 1

#     print('ac, tp, fp, fn', class_ac, class_tp, class_fp, class_fn)
    '''
    for i in range(labels.shape[1]):
        accuracy_list[i] = class_ac[i]/labels.shape[0]
        
        if class_tp[i] + class_fp[i] == 0:
            precision_list[i] = 0
        else:
            precision_list[i] = class_tp[i]/float((class_tp[i] + class_fp[i]))
        
        if class_tp[i] + class_fn[i] == 0:
            recall_list[i] = 0
        else:
            recall_list[i] = class_tp[i]/float((class_tp[i] + class_fn[i]))
    bcr_list = (precision_list + recall_list)/2.0 
    '''
    confusion_matrix = np.zeros((labels.shape[1]+1, labels.shape[1]+1))
    '''
    for i in range(labels.shape[0]):
        for num_iter in range(labels.shape[1]):
            if test_result[i, num_iter] == 0:
                if labels[i, num_iter] == 0:
                    confusion_matrix[0, 0] += 1
                else:
                    confusion_matrix[0, num_iter+1] += 1
            else:
                if labels[i, num_iter] == 0:
                    confusion_matrix[num_iter+1, 0] += 1
                else:
                    confusion_matrix[num_iter, 0] += 1
    '''           
    for i in range(labels.shape[0]):
        y = np.array(np.where(labels[i,:] == 1))
        x = np.array(np.where(test_result[i,:] == 1))        
        
        if x.shape[1] == 0 and y.shape[1] == 0:
            print("a")
            confusion_matrix[0,0] += 1 
            continue
        if x.shape[1] == 0:
            print("b")
            confusion_matrix[0,y+1] += 1 
            continue
        if y.shape[1] == 0:
            print("c")
            confusion_matrix[x+1,0] += 1 
            continue
        if x.shape[1] == 1 and y.shape[1] == 1:
            print("d")
            confusion_matrix[x+1,y+1] += 1 
            continue

        x_temp = set(list(x[0].astype(int)))
        y_temp = set(list(y[0].astype(int)))
        print(x_temp)
        print(y_temp)
        a = x_temp- (x_temp- y_temp)
        print(a)
        if len(a) == len(x_temp)== len(y_temp):
            print("e")
            for a_i in a:
                confusion_matrix[a_i+1,a_i+1] += 1
            continue
        if len(a) == 0:
            print("f")
            for i in x_temp:
                for j in y_temp:
                    confusion_matrix[i+1,j+1] += 1
            continue
        else:
            for a_i in a:
                confusion_matrix[a_i+1,a_i+1] += 1
            pre = y_temp -a
            tar = x_temp -a
            if len(tar) != 0 and len(pre) !=0:
                for m in pre:
                    for n in tar:
                        confusion_matrix[m+1,n+1] += 1
            if len(pre) == 0:
                for m in tar:
                    confusion_matrix[0,m+1] += 1
            if len(tar) == 0:
                for n in pre:
                    confusion_matrix[n+1,0] += 1
          
            
        #x = np.array(np.where(labels[i,:] == 1))
        #y = np.array(np.where(test_result[i,:] == 1))
        #for a, p in zip(test_result[i], labels[i]):
        #confusion_matrix[x+1,y+1] += 1
            
    return class_tp, class_fp, class_fn, class_ac,confusion_matrix
    #return accuracy_list, precision_list, recall_list, bcr_list
print(labels)
print(outputs>0.5)
prediction_perclass(labels, outputs)    


# In[74]:


x = np.array(np.where(np.array([1., 0., 0.]) == 1))


# In[75]:


x.shape


# In[79]:


def test(test_loader,model,direct):
    print('Resume model: %s' % direct)
    model_test = model.to(computing_device)
    check_point = torch.load(direct)
    model_test.load_state_dict(check_point['state_dict'])
    class_tp_all = np.zeros(14)
    class_fp_all = np.zeros(14)
    class_fn_all = np.zeros(14)
    class_ac_all = np.zeros(14)
    confusion_matrix_all = np.zeros((15,15))
 
    precision_class = np.zeros(14)
    recall_class = np.zeros(14)
    
    size = 0
    for mb_count, (test_images, test_labels) in enumerate(test_loader, 0):
        with torch.no_grad():      
            test_images, test_labels = test_images.to(computing_device), test_labels.to(computing_device)
            outputs = model_test(test_images)
            outputs = torch.sigmoid(outputs)
            class_tp, class_fp, class_fn,class_ac,confusion_matrix = prediction_perclass(test_labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            class_tp_all += class_tp
            class_fp_all += class_fp
            class_fn_all += class_fn
            class_ac_all += class_ac
            confusion_matrix_all += confusion_matrix
        print(mb_count)
        size = size + test_labels.shape[0]
            
    for i in range(14):
        if (class_tp_all[i] + class_fp_all[i]) == 0:
            precision_class[i]=0
        else:
            precision_class[i] = 1.0*class_tp_all[i]/(class_tp_all[i] + class_fp_all[i])
        if (class_tp_all[i] + class_fn_all[i]) == 0:
            recall_class[i]=0
        else:
            recall_class[i] = 1.0*class_tp_all[i]/(class_tp_all[i] + class_fn_all[i])
    
    accuracy_class = 1.0*class_ac_all/size
    precision_overall = 1.0*sum(class_tp_all)/(sum(class_tp_all) + sum(class_fp_all))
    recall_overall = 1.0*sum(class_tp_all)/(sum(class_tp_all) + sum(class_fn_all))
    
    accuracy_overal = 1.0*sum(class_ac_all)/14/size
    BCR_class = (precision_class + recall_class)/2.0
    BCR_overall = (precision_overall+recall_overall)/2.0
#     print(test_labels.shape[0])
    

    precision_class_1 = np.array(precision_class)
    np.save('precision_class', precision_class_1)
                    
    accuracy_class_1 = np.array(accuracy_class)
    np.save('accuracy_class', accuracy_class_1)
    
    accuracy_overal_1 = np.array(accuracy_overal)
    np.save('accuracy_overal', accuracy_overal_1)
    
    precision_overall_1 = np.array(precision_overall)
    np.save('precision_overall', precision_overall_1)
    
    recall_class_1 = np.array(recall_class)
    np.save('recall_class', recall_class_1)
    
    recall_overall_1 = np.array(recall_overall)
    np.save('recall_overall', recall_overall_1)
    
    BCR_class_1 = np.array(BCR_class)
    np.save('BCR_class',BCR_class_1)
    
    BCR_overall_1 = np.array(BCR_overall)
    np.save('BCR_overall',BCR_overall_1)  
    
    confusion_matrix_all_1 = np.array(confusion_matrix_all)
    np.save('confusion_matrix_all',confusion_matrix_all_1)  
    
    
    print('accuracy_class ',accuracy_class)
    print('accuracy_overal ',accuracy_overal)
    print('precision_class ',precision_class)
    print('precision_overall ',precision_overall)
    print("recall_class ",recall_class)    
    print("recall_overall ",recall_overall)
    print("BCR_class ",BCR_class)
    print("BCR_overall ",BCR_overall)
    print(confusion_matrix_all)
    


# In[80]:


from my_nn import *
from my_nn import MyCNN

direct = './cp_1/0_model_epoch800.pth'

# Setup: initialize the hyperparameters/variables
num_epochs = 1           # Number of full passes through the dataset
batch_size = 16          # Number of samples in each minibatch
learning_rate = 0.001  
seed = np.random.seed(1) # Seed the random number generator for reproducibility
p_val = 0.1              # Percent of the overall dataset to reserve for validation
p_test = 0.2             # Percent of the overall dataset to reserve for testing

#TODO: Convert to Tensor - you can later add other transformations, such as Scaling here
transform = transforms.Compose([transforms.Resize([128,128], interpolation=2),
                                #transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
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

# Setup the training, validation, and testing dataloaders
train_loader, val_loader, test_loader = create_split_loaders(batch_size, seed, transform=transform, 
                                                             p_val=p_val, p_test=p_test,
                                                             shuffle=True, show_sample=False, 
                                                             extras=extras)


model = MyCNN()
model = model.to(computing_device)
test(test_loader,model,direct)


# In[ ]:





# In[ ]:





# In[ ]:




