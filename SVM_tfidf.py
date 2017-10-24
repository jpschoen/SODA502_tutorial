# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:03:42 2017

@author: johnpschoeneman
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


#read in  Data and list features
ht_df = pd.read_csv("reddit_data/reddit_1006_test.csv")
ht_df = ht_df[ht_df.freq >= 43]

print(len(ht_df.author.unique()))

posts = ht_df['body']
digits = ht_df['author']

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(posts)
print(vectors.shape)


X_train, X_test, y_train, y_test = train_test_split(vectors, digits, test_size=0.2, random_state=19)


print(X_train.shape, X_test.shape)


svm = LinearSVC()
svm.fit(X_train, y_train)
 

predictions = svm.predict(X_test)
print(list(predictions[0:10]))

print(y_test[:10])


print(accuracy_score(y_test, predictions))


