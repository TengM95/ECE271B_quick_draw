#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
disease_classes = {0: "Atelectasis", 1: "Cardiomegaly", 2: "Effusion", 
                3: "Infiltration", 4: "Mass", 5: "Nodule", 6: "Pneumonia", 
                7: "Pneumothorax", 8: "Consolidation", 9: "Edema", 
                10: "Emphysema", 11: "Fibrosis", 
                12: "Pleural_Thickening", 13: "Hernia",14:"No Finding" }
disease_list = []
disease_list.append(disease_classes[14])
for i in range(14):
    disease_list.append(disease_classes[i])
print(disease_list)
# print(type(confusion_matrix))


# In[ ]:





# In[2]:


accuracy_class = np.load('accuracy_class.npy')


# In[3]:


accuracy_class = np.load('accuracy_class.npy')
accuracy_class = list(map(float, accuracy_class))
accuracy_class = ["%.3f" % member for member in accuracy_class]
accuracy_class = list(map(float, accuracy_class))
print('accuracy_class', accuracy_class)


# In[4]:


accuracy_overal = np.load('accuracy_overal.npy')
accuracy_overal = "%.3f" %accuracy_overal
print(accuracy_overal)


# In[5]:



precision_class = np.load('precision_class.npy')
precision_class = list(map(float, precision_class))
precision_class = ["%.3f" % member for member in precision_class]
precision_class = list(map(float, precision_class))
print('precision_class', precision_class)

precision_overall = np.load('precision_overall.npy')
precision_overall = "%.3f" %precision_overall

print(precision_overall)


# In[6]:


recall_class = np.load('recall_class.npy')

recall_class = list(map(float, recall_class))
recall_class = ["%.3f" % member for member in recall_class]
recall_class = list(map(float, recall_class))
print('recall_class', recall_class)

recall_overall = np.load('recall_overall.npy')
recall_overall = "%.3f" %recall_overall

print(recall_overall)

BCR_class = np.load('BCR_class.npy')

BCR_class = list(map(float, BCR_class))
BCR_class = ["%.3f" % member for member in BCR_class]
BCR_class = list(map(float, BCR_class))
print('BCR_class', BCR_class)

BCR_overall = np.load('BCR_overall.npy')
print(BCR_overall)


# In[7]:


BCR_overall = "%.3f" %BCR_overall
print(BCR_overall)


# In[8]:


confusion_matrix = np.load('confusion_matrix_all.npy')


# In[9]:


print(confusion_matrix)


# In[10]:


import numpy as np
# confusion_matrix 
sum_col = np.sum(confusion_matrix, axis = 0)
for i in range(15): 
    for j in range(15):
        confusion_matrix[i][j] = "{:.2g}".format(1.0*confusion_matrix[i][j]/sum_col[j])
        
print(confusion_matrix)


# In[11]:


string_c_m = []
short_disease_list = ['NoFind', 'Atelec', 'Cardi', 'Effus', 'Infil', 
                      'Mass', 'Nodul', 'Pneum', 'Pnem', 'Conso', 'Edema', 'Emphy', 'Fibro', 'Pleur', 'Herni']
for i in range(len(confusion_matrix)):
    temp = [str(i) for i in confusion_matrix[i]]
#     temp_1 = [str(i) for i in temp]
    string_c_m.append(list([str(short_disease_list[i])])+ temp)
print(string_c_m)


# In[12]:


from prettytable import PrettyTable

disease_new_list = list((disease_list[1:]) + ['Overall'])
# print(len(disease_new_list[0]))

temp_list = []
per = PrettyTable()

accuracy_class = [str(i) for i in accuracy_class]
precision_class = [str(i) for i in precision_class]
recall_class = [str(i) for i in recall_class]
BCR_class = [str(i) for i in BCR_class]

result_list = [[] for i in range(15)]

for i in range(14):   
    result_list[i].append(disease_list[i+1])
    result_list[i].append(accuracy_class[i]) 
    result_list[i].append(precision_class[i]) 
    result_list[i].append(recall_class[i]) 
    result_list[i].append(BCR_class[i])

result_list[14].append(disease_new_list[14])
result_list[14].append(str(accuracy_overal))
result_list[14].append(str(precision_overall))
result_list[14].append(str(recall_overall))
result_list[14].append(str(BCR_overall))

# print(result_list)

# print(type(accuracy_class[0]))
per.field_names = [" ", "Accuracy", "Precision", "Recall", "BCR"]
for i in range(len(disease_list)):
    per.add_row(result_list[i])

print(per)
    


# In[13]:


from prettytable import PrettyTable
    
x = PrettyTable()
d = [' ','NF', 'At', 'Ca' ,'Ef' , 'In' , 'Ma' , 'No' , 'Pnn','Pnt','Co','Ed','Em','Fi','PT','He']
# print(len(string_c_m[0]))
x.field_names = d
x.left_padding_width = 0
x.right_padding_width = 0
for i in range (15):
    x.add_row(string_c_m[i])
    
print(x)


# In[14]:


import numpy as np 
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import statistics
from os import listdir
from PIL import Image
def display_loss(loss_train,loss_vali):
    epc = list(range(len(loss_train)))
    epc = [x + 1 for x in epc]
    plt.plot(epc, loss_train, label='loss_train')
    plt.plot(epc, loss_vali, label='loss_validation')
    plt.title('Losses versus training minibatch at learning rate AC2')
    plt.xlabel('Number of minibatcbh')
    plt.ylabel('Loss over minibatcbh')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol=1)
    plt.show()
loss_val = np.load('avg_minibatch_loss-Copy1.npy')
loss_train = np.load('total_vali_loss-Copy1.npy')
display_loss(loss_train,loss_val[0:-1])


# In[ ]:




