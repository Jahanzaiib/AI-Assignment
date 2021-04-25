#!/usr/bin/env python
# coding: utf-8

# # Assignment: Ionosphere Data Problem
# 
# ### Dataset Description: 
# 
# This radar data was collected by a system in Goose Bay, Labrador. This system consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts. See the paper for more details. The targets were free electrons in the ionosphere. "Good" radar returns are those showing evidence of some type of structure in the ionosphere. "Bad" returns are those that do not; their signals pass through the ionosphere.
# 
# Received signals were processed using an autocorrelation function whose arguments are the time of a pulse and the pulse number. There were 17 pulse numbers for the Goose Bay system. Instances in this databse are described by 2 attributes per pulse number, corresponding to the complex values returned by the function resulting from the complex electromagnetic signal.
# 
# ### Attribute Information:
# 
# - All 34 are continuous
# - The 35th attribute is either "good" or "bad" according to the definition summarized above. This is a binary classification task.
# 
#  <br><br>
# 
# <table border="1"  cellpadding="6">
# 	<tbody>
#         <tr>
# 		<td bgcolor="#DDEEFF"><p class="normal"><b>Data Set Characteristics:&nbsp;&nbsp;</b></p></td>
# 		<td><p class="normal">Multivariate</p></td>
# 		<td bgcolor="#DDEEFF"><p class="normal"><b>Number of Instances:</b></p></td>
# 		<td><p class="normal">351</p></td>
# 		<td bgcolor="#DDEEFF"><p class="normal"><b>Area:</b></p></td>
# 		<td><p class="normal">Physical</p></td>
#         </tr>
#      </tbody>
#     </table>
# <table border="1" cellpadding="6">
#     <tbody>
#         <tr>
#             <td bgcolor="#DDEEFF"><p class="normal"><b>Attribute Characteristics:</b></p></td>
#             <td><p class="normal">Integer,Real</p></td>
#             <td bgcolor="#DDEEFF"><p class="normal"><b>Number of Attributes:</b></p></td>
#             <td><p class="normal">34</p></td>
#             <td bgcolor="#DDEEFF"><p class="normal"><b>Date Donated</b></p></td>
#             <td><p class="normal">N/A</p></td>
#         </tr>
#      </tbody>
#     </table>
# <table border="1" cellpadding="6">	
#     <tbody>
#     <tr>
# 		<td bgcolor="#DDEEFF"><p class="normal"><b>Associated Tasks:</b></p></td>
# 		<td><p class="normal">Classification</p></td>
# 		<td bgcolor="#DDEEFF"><p class="normal"><b>Missing Values?</b></p></td>
# 		<td><p class="normal">N/A</p></td>
# 		<td bgcolor="#DDEEFF"><p class="normal"><b>Number of Web Hits:</b></p></td>
# 		<td><p class="normal">N/A</p></td>
# 	</tr>
#     </tbody>
#     </table>

# ### WORKFLOW :
# - Load Data .

# In[289]:


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[290]:


pd.__version__


# In[291]:


np.__version__


# # Load Data:
# [Click Here to Download DataSet](https://github.com/ramsha275/ML_Datasets/blob/main/ionosphere_data.csv)

# Loading data...

# In[292]:


# Load the dataset.
df = pd.read_csv('ionosphere_data.csv', delimiter=',')


# In[293]:


# Find the shape of the dataset 
df.shape


# It's clear from the shape of the data that dataset is not a huge one. Only 351 records are available with 34 features/columns.

# In[294]:


df.head()


# In[ ]:





# In[295]:


df.describe().T


# In[296]:


for feature in df:
    print(feature)
    print(len(df[feature].unique()))


# In[297]:


df['feature2'].unique()


# In[298]:


df.drop(df.columns[1], inplace=True, axis=1)


# In[299]:


df.head()


# In[ ]:





# In[300]:


df.ndim


# In[301]:


df.info()


# In[302]:


'''for feature in df:
    print(feature)
    df[feature].hist()
    plt.show()'''

# df.hist()
# plt.show()


# In[303]:


# Check summary statistics
df.describe()


# - Check Missing Values ( If Exist ; Fill each record with mean of its feature ) or any usless column.

# In[304]:


# Find missing values
df.isnull().sum()


# In[305]:


df['label'] = [1 if lbl == 'g' else 0 for lbl in df['label']]


# In[306]:


train_data = df.sample(frac= 0.6, random_state=125)
test_data = df.drop(train_data.index)


# In[307]:


train_label = train_data.iloc[:,-1]
train_data = train_data.iloc[:,0:-1]
test_label = test_data.iloc[:,-1]
test_data = test_data.iloc[:,0:-1]


# In[308]:


# df.drop(columns= 'label', inplace = True)


# In[309]:


train_data.head()


# In[310]:


train_label


# - Standardized the Input Variables. **Hint**: Centeralized the data

# In[ ]:





# In[311]:


# # Normalize the data
# train_mean = train_data.mean()
# train_data -= train_mean
# train_std = train_data.std()
# train_data /= train_std
# test_data -= train_mean
# test_data /= train_std


# In[ ]:





# - Encode labels.

# - Shuffle the data if needed.
# - Split into 60 and 40 ratio.

# In[312]:


# Now sample the dataframe


# In[313]:


train_data.shape


# In[314]:


test_data.shape


# In[315]:


train_label.shape


# In[316]:


test_label.shape


# In[317]:


train_label.sum()


# In[318]:


len(train_label)


# In[319]:


# train_label.sum()/len(train_label)


# ### Data Preprocessing

# In[320]:


train_data = train_data.to_numpy()


# In[321]:


train_label = train_label.to_numpy().astype('float32')


# In[322]:


test_data = test_data.to_numpy()


# In[323]:


test_label = test_label.to_numpy().astype('float32')


# In[324]:


#train_set = np.array(train_set.as_matrix())
#train_label = np.array(pd.DataFrame(train_label).as_matrix())


# In[325]:


print(type(train_data))
print(type(train_label))
print(type(test_data))
print(type(test_label))


# In[326]:


print(train_data.dtype)
print(train_label.dtype)
print(test_label.dtype)
print(test_data.dtype)


# ### Model Architecture

# - Model : 1 hidden layers including 16 unit.

# In[327]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,  activation='sigmoid'))


# In[328]:


model.summary()


# - Compilation Step (Note : Its a Binary problem , select loss , metrics according to it)

# In[329]:


from tensorflow.keras import optimizers

model.compile(optimizer = 'RMSprop', loss='binary_crossentropy', metrics=['accuracy'])


# - Train the Model with Epochs (100).

# In[330]:


history = model.fit(train_data, train_label, validation_split=0.2, epochs=75, batch_size = 16)


# In[331]:


history.history.keys()


# In[332]:


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(75)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[333]:


acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# - Evaluation Step
# - Prediction

# In[334]:


score = model.evaluate(test_data, test_label)


# In[335]:


score


# - If the model gets overfit tune your model by changing the units , No. of layers , epochs , add dropout layer or add Regularizer according to the need .
# - Prediction should be > **92%**

# In[336]:


predictions=model.predict(test_data)


# In[337]:


y_pred = (predictions > 0.5)


# In[338]:


tf.math.confusion_matrix(
    test_label, y_pred, num_classes=2, weights=None, dtype=tf.dtypes.int32,
    name=None
)


# In[339]:


# It will evaluate the logical expression y_predict>0.25 and return True or False 


# In[340]:


np.count_nonzero(y_pred)

