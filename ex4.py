#!/usr/bin/env python
# coding: utf-8

# ##### Daniil Rolnik 334018009

# 
# ___
# # Decision Trees

# 
# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


# ## Global Variables

# In[2]:


label_column_name = 'label'
d = defaultdict(LabelEncoder)


# ## Get the Data

# In[3]:


headers = ['checking_status', 'saving_status', 'credit_history', 'housing', 'job', 'property_magnitude',
                'number_of_dependents', 'number_of_existing_credits', 'own_telephone', 'foreign_workers', 'label']
label_column_name = 'label'
d = defaultdict(LabelEncoder)
dataset = pd.read_csv("dataset/train.txt", header=None, names=headers)


# In[4]:


df = pd.DataFrame(dataset,columns=headers)


# In[5]:


df.head()


# In[6]:


dataset_test = pd.read_csv("dataset/test.txt", header=None, names=headers)
df_test = pd.DataFrame(dataset,columns=headers)


# In[7]:


df_test.head()


# ## Train Test Split
# 
# Split up the data into a training set and a test set!

# In[8]:


y_train = df[label_column_name]
df = df.drop(label_column_name,axis=1)
X_train = df
y_test = df_test[label_column_name]

df_test = df_test.drop(label_column_name,axis=1)
X_test = df_test
print(f'X_train num of features = {len(X_train.columns)}')
print(f'X_test num of features = {len(X_test.columns)}')
X_train.head()
X_test.head()


# ## Decision Trees
# 
# We'll start just by training a single decision tree.

# In[9]:


from sklearn.tree import DecisionTreeClassifier


# In[10]:


dtree = DecisionTreeClassifier(criterion = 'entropy')


# In[11]:


# Encoding the variable
X_train = df.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
X_train.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
df.apply(lambda x: d[x.name].transform(x))

# Encoding the variable
X_test = df.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
X_test.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
df.apply(lambda x: d[x.name].transform(x))


# In[12]:


dtree.fit(X_train,y_train)


# ## Prediction and Evaluation 
# 
# Let's evaluate our decision tree.

# In[13]:


predictions = dtree.predict(X_test)


# In[ ]:





# In[14]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[15]:


conf_matrix=confusion_matrix(y_test,predictions)
accuracy=accuracy_score(y_test,predictions)


# In[16]:


#### print decision tree accuracy
conf_matrix,accuracy


# In[17]:


print(classification_report(y_test,predictions))


# In[18]:


print(confusion_matrix(y_test,predictions))


# ## Tree Visualization
# 

# In[19]:


from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot 

features = list(df.columns[:])
features


# ### Full Tree visualization

# In[20]:


dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True, class_names=['B','G'])


# In[21]:


graph = pydot.graph_from_dot_data(dot_data.getvalue())  
print(graph)
graph[0].write_png('full-decision_tree.png')
Image(graph[0].create_png())  


# ### Final Tree visualization

# In[22]:


dtree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
dtree.fit(X_train,y_train)
dot_data = StringIO()  
export_graphviz(dtree, out_file=dot_data,feature_names=features,filled=True,rounded=True, class_names=['B','G'])
graph = pydot.graph_from_dot_data(dot_data.getvalue())  
print(graph)
graph[0].write_png('final_decision_tree.png')
Image(graph[0].create_png())

