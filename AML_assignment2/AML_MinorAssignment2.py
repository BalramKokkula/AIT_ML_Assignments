import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups


from sklearn.feature_extraction.text import TfidfVectorizer

def tokenize(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

news = fetch_20newsgroups(subset='all')

x, y = news.data, news.target


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=32)

#### SVM Classification Model ######

svc_clf = Pipeline([('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'), tokenizer=tokenize)), ('classifier', LinearSVC())])
svc_clf.fit(x_train, y_train)
print(svc_clf.score(x_test, y_test))
y_pred = svc_clf.predict(x)
svc_cm = pd.DataFrame(confusion_matrix(y, y_pred))

print("SVC Model Acuuracy score:",svc_clf.score(x_test, y_test))
print("Confusion matrix of SVC Model:")
print(svc_cm)


#### Naive Bayes Model #######

nb_clf = Pipeline([('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'), tokenizer=tokenize)), ('classifier', MultinomialNB(alpha=0.005))])
nb_clf.fit(x_train, y_train)
y_pred = nb_clf.predict(x)
nb_cm = pd.DataFrame(confusion_matrix(y, y_pred))

print("Naive Bayes Model Acuuracy score:",nb_clf.score(x_test, y_test))
print("Confusion matrix of Naive Bayes Model:")
print(nb_cm)


#### Neural Network Model ######

nn_clf = Pipeline([('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'), tokenizer=tokenize)), ('classifier', MLPClassifier(hidden_layer_sizes=(20,10), max_iter=10))])
nn_clf.fit(x_train, y_train)
y_pred = nn_clf.predict(x)
nn_cm = pd.DataFrame(confusion_matrix(y, y_pred))

print("Neural Network Model Acuuracy score:",nn_clf.score(x_test, y_test))
print("Confusion matrix of Neural Network Model:")
print(nn_cm)




