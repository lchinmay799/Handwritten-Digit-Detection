#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm


# In[2]:


from torchvision import datasets,transforms
mnist_train=datasets.MNIST(root='./datasets',train=True,transform=transforms.ToTensor(),download=True)
mnist_test=datasets.MNIST(root='./datasets',train=False,transform=transforms.ToTensor(),download=False)


# In[3]:


train_data=torch.utils.data.DataLoader(mnist_train,shuffle=True,batch_size=100)
test_data=torch.utils.data.DataLoader(mnist_test,shuffle=False,batch_size=100)


# In[4]:


import torch.nn as nn


# In[5]:


class logistic(nn.Module):
    def __init__(self,inp,out):
        super().__init__()
        self.lin=nn.Linear(inp,out)
    def forward(self,x):
        return self.lin(x)


# In[6]:


model=logistic(784,10)


# In[11]:


optimizer=torch.optim.SGD(model.parameters(),lr=0.01)


# In[12]:


entropy=nn.CrossEntropyLoss()


# In[13]:


epoch=10
for i in range(epoch):
    for images,labels in tqdm(train_data):
        optimizer.zero_grad()
        x=images.view(-1,784)
        y=model(x)
        loss=entropy(y,labels)
        loss.backward()
        optimizer.step()


# In[14]:


correct,total=0,len(test_data)
for images,labels in tqdm(test_data):
    x=images.view(-1,784)
    y=model(x)
    prediction=torch.argmax(y,dim=1)
    correct+=torch.sum((prediction==labels).float())
print("Accuracy : ",correct/total)


# In[ ]:




