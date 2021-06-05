## collecting data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## reading file

df=pd.read_csv("news.csv")
df.head()


## analyzing data
sns.countplot(x='label',data=df);

## data wrangling
df.isnull()
df.isnull().sum()
df.drop("title",axis=1,inplace=True)
df

label=pd.get_dummies(df["label"])
label

label=pd.get_dummies(df["label"],drop_first=True)
label

df=pd.concat([df,label],axis=1)
df
df.drop("label",axis=1,inplace=True)
df


## test and training data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['REAL'], test_size=0.7, random_state=7)

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english')

#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred)







