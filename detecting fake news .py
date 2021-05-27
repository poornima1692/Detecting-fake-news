#!/usr/bin/env python
# coding: utf-8

# In[73]:



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[74]:


df=pd.read_csv("news.csv")


# In[75]:


df.head()


# In[76]:


sns.countplot(x='label',data=df);


# In[77]:


df.isnull()


# In[78]:


df.isnull().sum()


# In[79]:


df.drop("title",axis=1,inplace=True)


# In[80]:


df


# In[81]:


label=pd.get_dummies(df["label"])


# In[82]:


label


# In[83]:


label=pd.get_dummies(df["label"],drop_first=True)


# In[84]:


label


# In[85]:


df=pd.concat([df,label],axis=1)


# In[86]:


df


# In[87]:


df.drop("label",axis=1,inplace=True)


# In[88]:


df


# In[89]:


from sklearn.model_selection import train_test_split


# In[90]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[91]:


#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['REAL'], test_size=0.7, random_state=7)


# In[92]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[104]:


#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english')

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[105]:


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[106]:


#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[107]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred)


# In[ ]:




