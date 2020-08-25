import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import tweepy


from wordcloud import WordCloud, STOPWORDS
from tweepy import OAuthHandler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.neural_network import MLPClassifier




##To connect to the Twitter Application server from a Python client.
##use the consumer API key, consumer API secret, Access token, and Access token secret.

consumer_key = 'L7goSyjrhAihsOL7OkEWRYmzM'
consumer_secret = 'j0dYOE8rjotYHPjTLmwWjg0panXB7wckylVBD3ZUzSg6DBY5ua'
access_token = '223816512-NvtdHkJAvBL907cmEyMAyQgUpw8Jmf6EdKNzi1Sj'
access_token_secret = 'Q1mn4toX3RgiT68fgM5ptrN6Pe3okvWDzcSpiXgXj4cqI'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# Creating Twitter List

def list_tweets(user_id, count, prt=False):
    tweets = api.user_timeline(
        "@" + user_id, count=count, tweet_mode='extended')
    tw = []
    for t in tweets:
        tw.append(t.full_text)
        if prt:
            print(t.full_text)
            print()
    return tw

# trump user name in twitter.
user_id = 'realDonaldTrump'
count=200



#Cleaning Twitter Dataset


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def clean_tweets(lst):
    # remove twitter Return handles (RT @xxx:)
    lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
    # remove URL links (httpxxx)
    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
    return lst


# Performing vader sentiment analysis

analyser = SentimentIntensityAnalyzer()
analyser.polarity_scores("The movie is good")


def sentiment_analyzer_scores(text):
    score = analyser.polarity_scores(text)
    lb = score['compound']
    if lb >= 0.05:
        return 1
    elif (lb > -0.05) and (lb < 0.05):
        return 0
    else:
        return -1

sentiment_analyzer_scores("The movie is VERY BAD!!!")

    
# analyzing and plotting tweets of @ tweets sentiment. 
def anl_tweets(lst, title='Tweets Sentiment', engl=True ):
    sents = []
    for tw in lst:
        st = sentiment_analyzer_scores(tw)
        sents.append(st)
    ax = sns.distplot(
        sents,
        kde=False,
        bins=3)
    ax.set(xlabel='Negative          Neutral                 Positive',
           ylabel='#Tweets',
          title="Tweets of @"+title)
    return sents    

#Visualization of dataset with WordCloud
def word_cloud(wd_list):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=21,
        colormap='jet',
        max_words=50,
        max_font_size=200).generate(all_words)

    plt.figure(figsize=(12, 10))
    plt.axis('on')
    plt.imshow(wordcloud, interpolation="bilinear")
    
tw_trump = list_tweets(user_id, count)
tw_trump = clean_tweets(tw_trump)
tw_trump_sent = anl_tweets(tw_trump)
type(tw_trump)
tw_trump[2]
sentiment_analyzer_scores(tw_trump[19])
word_cloud(tw_trump)



# creating tweets dataframe, train and test data

df = pd.DataFrame(tw_trump,columns =['tweets'])
df['tweets'] =  clean_tweets(df['tweets'])
df['sent'] = anl_tweets(df.tweets)

print(df.head(10))

# df.to_csv("trump_tweeets.csv")


X = df['tweets']
y = df['sent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)


# This is a tokenizer that will be used when transforming the message to a Bag of Words
# creating tfidf train and test data

def Tokenizer(str_input):
    str_input = str_input.lower()
    words = word_tokenize(str_input)
    #remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    #stem the words
    porter_stemmer=nltk.PorterStemmer()
    words = [porter_stemmer.stem(word) for word in words]
    return words

vectorizer = TfidfVectorizer(tokenizer=Tokenizer, min_df=0.001, max_df=0.3)
X_tfidf_train = vectorizer.fit_transform(X_train)
X_tfidf_test = vectorizer.transform(X_test)


# SVM Classifier Model

svc_clf = SVC(kernel='linear')
svc_clf.fit(X_tfidf_train,y_train)
predictions = svc_clf.predict(X_tfidf_test)

print("SVM Accuracy:", accuracy_score(y_test, predictions))
print("SVM Precision:", precision_score(y_test, predictions, average='weighted'))


#Naive Bayes Model

nb_clf = naive_bayes.MultinomialNB()
nb_clf.fit(X_tfidf_train,y_train)
nb_predictions = nb_clf.predict(X_tfidf_test)

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print("Naive Bayes Precision:", precision_score(y_test, nb_predictions, average='weighted'))


# Neural Network Model

nn_clf = MLPClassifier(hidden_layer_sizes=(200,100))
nn_clf.fit(X_tfidf_train,y_train)
nn_predictions = nn_clf.predict(X_tfidf_test)

print("neural_network Accuracy:", accuracy_score(y_test, nn_predictions))
print("neural_network Precision:", precision_score(y_test, nn_predictions, average='weighted'))