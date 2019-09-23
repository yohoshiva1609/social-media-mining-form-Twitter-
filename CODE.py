#step-01
#importing requests in packages
import tweepy
import re
import pickle
from tweepy import OAuthHandler
from nltk.corpus import stopwords


#step-02
consumer_key='AlLzcJ0wODn13AainIIyHor6K'
consumer_secret='khDWWSQlYgCNcCardctioCWxvY81go0JAeI2RAAp4dIkrayaA6'
access_token='939359239125942272-uL5vCEu4zwfroICqYwdB1ZE6XfjFV3l'
access_secret='7r7jELl62x49EkzqSCIiSVXaQuKGupNtNCxkpOYv8XFqu'

auth=OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)



args=['tiktok']

api=tweepy.API(auth,timeout=10)

list_tweets=[]

query=args[0]
if len(args)==1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent').items(100):
        list_tweets.append(status.text)


#loading pickle file 
with open('TfidfVectorizer.pickle','rb') as f:
    tfidf=pickle.load(f)

with open('LogisticRegression.pickle','rb') as f:
    clf=pickle.load(f)

   
total_pos=0
total_neg=0

#preprocessing 
for tweet in list_tweets:
    tweet=re.sub(r"^http://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet=re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s"," ",tweet)
    tweet=re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$"," ",tweet)
    tweet=tweet.lower()
    tweet=re.sub(r"that's","that is",tweet)
    tweet=re.sub(r"there's","there is",tweet)
    tweet=re.sub(r"what's","what is",tweet)
    tweet=re.sub(r"where's","where is",tweet)
    tweet=re.sub(r"it's","it is",tweet)
    tweet=re.sub(r"who's","who is",tweet)
    tweet=re.sub(r"i'm","i am",tweet)
    tweet=re.sub(r"she's","she is",tweet)
    tweet=re.sub(r"he's","he is",tweet)
    tweet=re.sub(r"they're","they are",tweet)
    tweet=re.sub(r"who're","who are",tweet)
    tweet=re.sub(r"ain't","am not",tweet)
    tweet=re.sub(r"wouldn't","would not",tweet)
    tweet=re.sub(r"should'n","should not",tweet)
    tweet=re.sub(r"can't","can not",tweet)
    tweet=re.sub(r"could't","could not",tweet)
    tweet=re.sub(r"\W"," ",tweet)
    tweet=re.sub(r"\d"," ",tweet)
    tweet=re.sub(r"s+[a-z]\s+","",tweet)
    tweet=re.sub(r"\s+[a-z]$","",tweet)
    tweet=re.sub(r"^[a-z]\s+"," ",tweet)
    tweet=re.sub(r"\s+"," ",tweet)
    #print(tweet)
    sent=clf.predict(tfidf.transform([tweet]).toarray())
    #print(tweet,":",sent)
    if sent[0]==1:
        total_pos+=1
    else:
        total_neg+=1

import matplotlib.pyplot as mp
import numpy as np
objects=['Positive','Negative']
y_pos=np.arange(len(objects))
mp.bar(y_pos,[total_pos,total_neg],alpha=0.5)
#mp.xticks(y_pos,objects)
mp.ylabel('Number')
mp.title("Number of positive and Negative tweets")
mp.show()

















