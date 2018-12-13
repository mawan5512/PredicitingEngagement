#!/usr/bin/env python
# coding: utf-8

# In[136]:


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.metrics import mean_squared_log_error
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
tknzr = TweetTokenizer()
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
TaggedDocument = gensim.models.doc2vec.TaggedDocument
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")


# In[56]:


tweets_data_path = '/Users/mohammedawan/Downloads/School/2018-2 Fall/Info Retrieval:Knowledge Discovery [CSI 5810]/Project 2/twitter_data.txt'

tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue


# In[57]:


i = 0
retweets = []
users = []
for tweet in tweets_data:
    if('retweeted_status' in tweet):
        retweets.append(tweet['retweeted_status'])
    i = i + 1
for tweet in retweets:
    users.append(tweet['user'])


# In[58]:


print(len(tweets_data))


# In[59]:


text = []
text_length = []
lang = []
country = []
engage = []
follower = []
activity = []
listed = []
verified = []
hashtag_num = []
i = 0
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"   
                           u"\U0001F927"
                           "]+", flags=re.UNICODE)
for tweet, user in zip(retweets, users):
    if('extended_tweet'in tweet):
        ex_tweet = tweet['extended_tweet']
        text.append(emoji_pattern.sub(r'', ex_tweet['full_text']))
        text_length.append(len(ex_tweet['full_text']))
    else:
        text.append(emoji_pattern.sub(r'', tweet['text']))
        text_length.append(len(tweet['text']))
    lang.append(tweet['lang'])
    engage.append(tweet['retweet_count'] + tweet['favorite_count'] + tweet['reply_count'])
    follower.append(user['followers_count'] + user['friends_count'])
    activity.append(user['favourites_count'] + user['statuses_count'])
    listed.append(user['listed_count'])
    verified.append(user['verified'])
    hashtag_num.append(len(tweet['entities']['hashtags']) + len(tweet['entities']['user_mentions']) + len(tweet['entities']['urls']))
lb.fit(lang)
binlang = lb.transform(lang)


# In[60]:


qtweets = pd.DataFrame.from_csv('/Users/mohammedawan/Downloads/School/2018-2 Fall/Info Retrieval:Knowledge Discovery [CSI 5810]/Project 2/sample.csv', encoding = "ISO-8859-1")
qguess = qtweets['Engagement']
qtext = qtweets['Text']
qact = qtweets['Activity']
qlength = qtweets['Length']
qent = qtweets['Entities']


# In[61]:


atweets = pd.DataFrame.from_csv('/Users/mohammedawan/Downloads/School/2018-2 Fall/Info Retrieval:Knowledge Discovery [CSI 5810]/Project 2/sample2.csv', encoding = "ISO-8859-1")
aguess = atweets['Engagement']
samp_ans = []


# In[138]:


samp_ans = []
q_ans = []
a_ans = []
samp = []
for t, e in zip(text, engage):
    for qt, qg, ag in zip(qtext, qguess, aguess):
        if (t == qt):
            samp.append(t)
            samp_ans.append(e)
            q_ans.append(qg)
            a_ans.append(ag)
from sklearn.metrics import r2_score
print("Person 1:")
print("R2 Score-" + str(r2_score(samp_ans, q_ans, multioutput='raw_values')))
print("Person 2:")
print("R2 Score-" + str(r2_score(samp_ans, a_ans, multioutput='raw_values')))


# In[63]:


def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized
tokenx = []
for t in text:
    tokenx.append(tknzr.tokenize(t))
tokenx = [word for word in tokenx if word not in stopwords.words('english')]
text_labels = labelizeTweets(tokenx, 'TEXT')


# In[73]:


tweet_w2v = Word2Vec(size=200, min_count=1)
tweet_w2v.build_vocab([x.words for x in tqdm(text_labels)])
tweet_w2v.train([x.words for x in tqdm(text_labels)], total_examples=200, epochs=20)


# In[66]:


w2v = []
for t in tokenx:
    tex = []
    for w in t:
        if w in tweet_w2v.wv.vocab.keys():
            tex.append(tweet_w2v[w])
    if not t:
        w2v.append([])
    else:
        w2v.append(tex)
w2vtext = []
count = 0
for t in w2v:
    if not t:
        w2vtext.append(np.array([0]*400))
    else:
        mini = np.array(t).min(axis = 0)
        maxi = np.array(t).max(axis = 0)
        w2vtext.append(np.array(np.concatenate((mini, maxi), axis = 0).tolist()))


# In[67]:


tweets = pd.DataFrame()
vec = pd.DataFrame(w2vtext)
lan = pd.DataFrame(binlang)
tweets = pd.concat([tweets,vec,lan], axis=1)
tweets['follower'] = follower
tweets['activity'] = activity
tweets['listed'] = listed
tweets['verified'] = verified
tweets['length'] = text_length
tweets['entities'] = hashtag_num
tweets['engagement'] = engage


# In[68]:


tweets


# In[69]:


tweets.to_csv('/Users/mohammedawan/Downloads/School/2018-2 Fall/Info Retrieval:Knowledge Discovery [CSI 5810]/Project 2/output.csv')


# In[ ]:




