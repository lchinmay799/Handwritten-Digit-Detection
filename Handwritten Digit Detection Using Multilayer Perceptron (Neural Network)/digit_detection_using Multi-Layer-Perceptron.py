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
mnist_train=datasets.MNIST(root='./datasets',train=True,transform=transforms.ToTensor(),download=True)
mnist_test=datasets.MNIST(root='./datasets',train=False,transform=transforms.ToTensor(),download=True)


# In[3]:


train_data=torch.utils.data.DataLoader(mnist_train,batch_size=100,shuffle=True)
test_data=torch.utils.data.DataLoader(mnist_test,batch_size=100,shuffle=True)


# In[4]:


import torch.nn as nn


# In[5]:


class MLP(nn.Module):
    def __init__(self,inp,hid1,hid2,out):
        super().__init__()
        self.ml=nn.Sequential(nn.Linear(inp,hid1),nn.ReLU(),
                             nn.Linear(hid1,hid2),nn.ReLU(),
                             nn.Linear(hid2,out))
    def forward(self,x):
        return self.ml(x)


# In[6]:


model=MLP(784,600,350,10)


# In[7]:


optimizer=torch.optim.SGD(model.parameters(),lr=0.1)


# In[8]:


entropy=nn.CrossEntropyLoss()


# In[9]:


epoch=10
for i in range(epoch):
    for images,labels in tqdm(train_data):
        optimizer.zero_grad()
        x=images.view(-1,784)
        y=model(x)
        loss=entropy(y,labels)
        loss.backward()
        optimizer.step()
    


# In[10]:


correct,total=0,len(test_data)
for images,labels in tqdm(test_data):
    x=images.view(-1,784)
    y=model(x)
    prediction=torch.argmax(y,dim=1)
    correct+=torch.sum((prediction==labels).float())
print("Accuracy : ",correct/total)


# In[ ]:




