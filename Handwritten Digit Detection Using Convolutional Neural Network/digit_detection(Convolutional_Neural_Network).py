#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


# In[2]:


from torchvision import datasets,transforms


# In[3]:


mnist_train=datasets.MNIST(root='./datasets',train=True,download=True,transform=transforms.ToTensor())
mnist_test=datasets.MNIST(root='./datasets',train=False,download=True,transform=transforms.ToTensor())


# In[4]:


train_data=torch.utils.data.DataLoader(mnist_train,batch_size=100,shuffle=True)
test_data=torch.utils.data.DataLoader(mnist_test,batch_size=100,shuffle=False)


# In[5]:


import torch.nn as nn
import torch.nn.functional as F


# In[6]:


class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution=nn.Sequential(nn.Conv2d(1,32,kernel_size=5,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2),
                                       nn.Conv2d(32,64,kernel_size=5,padding=2),nn.ReLU(),nn.MaxPool2d(kernel_size=2))
        self.mlp=nn.Sequential(nn.Linear(7*7*64,256),nn.ReLU(),
                               nn.Linear(256,10))
    def forward(self,x):
        x=self.convolution(x)
        x=x.view(-1,7*7*64)
        x=self.mlp(x)
        return x


# In[7]:


model=cnn()


# In[8]:


print(model)
print(len(list(model.parameters())))


# In[9]:


optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


# In[10]:


entropy=nn.CrossEntropyLoss()


# In[11]:


epoch=3
for i in range(epoch):
    for images,labels in tqdm(train_data):
        optimizer.zero_grad()
        y=model(images)
        loss=entropy(y,labels)
        loss.backward()
        optimizer.step()
    


# In[12]:


correct,total=0,len(test_data)
for images,labels in tqdm(test_data):
    y=model(images)
    prediction=torch.argmax(y,dim=1)
    correct+=torch.sum((prediction==labels).float())
print("Accuracy : ",correct/total)


# In[ ]:




