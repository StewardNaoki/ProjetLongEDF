#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[42]:


liste=[]# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[43]:



dfA1 = pd.read_csv('inputA1.csv',names=['a11', 'a12','a13','a14','a15','a16'])
dfA2 = pd.read_csv('inputA2.csv',names=['a21', 'a22','a23','a24','a25','a26'])
dfA3 = pd.read_csv('inputA3.csv',names=['a31', 'a32','a33','a34','a35','a36'])
dfA4 = pd.read_csv('inputA4.csv',names=['a41', 'a42','a43','a44','a45','a46'])
dfA5 = pd.read_csv('inputA5.csv',names=['a51', 'a52','a53','a54','a55','a56'])


dfC = pd.read_csv('inputC.csv',names=['c1', 'c2','c3','c4','c5','c6'])
dfB1 = pd.read_csv('inputB1.csv',names=['B1'])
dfB2 = pd.read_csv('inputB2.csv',names=['B2'])
dfB3 = pd.read_csv('inputB3.csv',names=['B3'])
dfB4 = pd.read_csv('inputB4.csv',names=['B4'])
dfB5 = pd.read_csv('inputB5.csv',names=['B5'])
target= pd.read_csv('output.csv',names=['Var0', 'Var1','Var2','Var3','Var4','Var5'])


# In[44]:


dataset = pd.concat([dfA1,dfA2,dfA3,dfA4,dfA5,dfB1,dfB2,dfB3,dfB4,dfB5,dfC,target], axis=1, sort=False)
train, validate, test = np.split(dataset.sample(frac=1), [int(.6*len(dataset)), int(.8*len(dataset))])
print(dataset)


# In[45]:


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(34, kernel_initializer='normal',input_dim = train[['a11', 'a12','a13','a14','a15','a16',
                                                                    'a21', 'a22','a23','a24','a25','a26',
                                                                    'a31', 'a32','a33','a34','a35','a36',
                                                                    'a41', 'a42','a43','a44','a45','a46',
                                                                     'a51', 'a52','a53','a54','a55','a56', 
                                                                    'B1','B2','B3','B4','B5',
                                                                    'c1', 'c2','c3','c4','c5','c6']].shape[1],
                                                                    activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(100, kernel_initializer='normal',activation='relu'))

#NN_model.add(Dense(30, kernel_initializer='normal',activation='relu'))


# The Output Layer :
NN_model.add(Dense(6, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
#NN_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[48]:


checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 2, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[41]:


NN_model.fit(train[['a11', 'a12','a13','a14','a15','a16',
                    'a21', 'a22','a23','a24','a25','a26',
                    'a31', 'a32','a33','a34','a35','a36',
                    'a41', 'a42','a43','a44','a45','a46',
                    'a51', 'a52','a53','a54','a55','a56',
                    'B1','B2','B3','B4','B5',
                    'c1', 'c2','c3','c4','c5','c6']], train[['Var0', 'Var1','Var2','Var3','Var4','Var5']], epochs=500,
                     batch_size=32, validation_split = 0.2, callbacks=callbacks_list,
                    )


# In[47]:


NN_model.fit(train[['a11', 'a12','a13','a14','a15','a16',
                    'a21', 'a22','a23','a24','a25','a26',
                    'a31', 'a32','a33','a34','a35','a36',
                    'a41', 'a42','a43','a44','a45','a46',
                    'a51', 'a52','a53','a54','a55','a56',
                    'B1','B2','B3','B4','B5',
                    'c1', 'c2','c3','c4','c5','c6']], train[['Var0', 'Var1','Var2','Var3','Var4','Var5']], epochs=500,
                     batch_size=32, validation_split = 0.2, callbacks=tensorboard_callback,
                    )


# In[9]:


print(callbacks_list)


# In[11]:



# Load wights file of the best model :
wights_file = 'Weights-440--2.36772.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# In[36]:



predictions = NN_model.predict(test[['a11', 'a12','a13','a14','a15','a16',
                                    'a21', 'a22','a23','a24','a25','a26',
                                    'a31', 'a32','a33','a34','a35','a36',
                                     'a41', 'a42','a43','a44','a45','a46',
                                     'a51', 'a52','a53','a54','a55','a56',
                                     'B1','B2','B3','B4','B5',
                                    'c1', 'c2','c3','c4','c5','c6']])
print(predictions[:,1][:])


# In[41]:


print(predictions[:,1][0])


# In[14]:


print(test[['Var0', 'Var1','Var2','Var3','Var4','Var5']])


# In[ ]:


#from keras.layers import Input, Dense
#from keras.models import Model

# This returns a tensor
#inputs = Input(shape=(4,))
# a layer instance is callable on a tensor, and returns a tensor
#output_1 = Dense(64, activation='relu')(inputs)
#output_2 = Dense(64, activation='relu')(output_1)
#predictions = Dense(1, activation='softmax')(output_2)

# This creates a model that includes
# the Input layer and three Dense layers
#model = Model(inputs=inputs, outputs=predictions)
#model.compile(optimizer='rmsprop',
#              loss='mean_absolute_error',
#              metrics=['accuracy'])
#model.fit(train, target)  # starts training


# In[42]:



plt.plot(test[['Var0', 'Var1','Var2','Var3','Var4','Var5']],'.')


# In[42]:


test.head(10)


# In[74]:


print(predictions[:,1][:])
print(test['Var1'][0])
print((predictions[:,0][0]-test['Var0'][0]))


# In[68]:


a=test['Var1'].to_numpy()
type(a)


# In[52]:


for i in range(200):
    a[i]=


# In[82]:


plt.hist(a)
plt.hist(predictions[:,1][:])


# In[84]:


for i in range(200):
    erreur=(a[i]-predictions[:,1][i])*(a[i]-predictions[:,1][i])
    print(erreur)


# In[ ]:




