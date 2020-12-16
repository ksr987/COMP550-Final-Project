# Importing libraries

import argparse
import numpy as np
import pandas as pd
import pickle
import sklearn
import nltk
from collections import Counter

# from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier , GradientBoostingClassifier , StackingClassifier , VotingClassifier

def train(data_path : str, num_ex : int, classifier : str, alpha : float):

  # Loading data from pickle file
  infile = open(data_path,'rb')
  train = pickle.load(infile)
  infile.close()

  advice = askreddit = coronavirus = twitterfeed = gaming = gardening = modernwarfare = \
  politics = relationship_advice = techsupport = []

  # Modifying data for training
  def listofreddits(reddit_name, list_name):
    list_name = [[] for i in range(num_ex)]
    for i,a in enumerate(train[reddit_name]):
      if i < num_ex:
        list_name[i] = " ".join(a[0]) + " " + " ".join(a[1])

    return list_name

  advice = listofreddits('Advice', advice)
  askreddit = listofreddits('AskReddit', askreddit)
  coronavirus = listofreddits('Coronavirus', coronavirus)
  twitterfeed = listofreddits('TheTwitterFeed', twitterfeed)
  gaming = listofreddits('gaming', gaming)
  gardening = listofreddits('gardening', gardening)
  modernwarfare = listofreddits('modernwarfare', modernwarfare)
  politics = listofreddits('politics', politics)
  relationship_advice = listofreddits('relationship_advice', relationship_advice)
  techsupport = listofreddits('techsupport', techsupport)

  # Building trainset
  print('Building trainset..')
  train_set = advice + askreddit + coronavirus + twitterfeed + gaming + gardening + modernwarfare + politics + relationship_advice + techsupport


  # Dataframe with text 
  train_data = pd.DataFrame()
  train_data['comments'] = train_set

  # Subreddits list
  subreddits = ['Advice','AskReddit','Coronavirus','TheTwitterFeed','gaming','gardening','modernwarfare','politics','relationship_advice','techsupport']

  # Dataframe with text and respective subreddit class
  title = []
  for s in subreddits:
    title += [s] * num_ex
  train_data['title'] = title

  # Creating a list of all reviews to be passed into vectorizer for feature extraction
  list_of_reviews = list(train_data['comments'])

  # tfidf vectorization
  vect = TfidfVectorizer()

  # Creating bag of words with features (unigram/ bigram/ .. ngrams)
  data = pd.DataFrame(vect.fit_transform(list_of_reviews).toarray(), index=list_of_reviews, columns= vect.get_feature_names()) # Creating bag of words with unigram features
  data.index = train_data.index
  # Adding label 
  data['label_or_Class'] = train_data['title']

  # Shuffling
  data = data.sample(frac=1)
  X = data.iloc[:,:-1] # X - bag of words (input)
  y = data.iloc[:,-1]  # y - labels

  # Splitting train, val
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=127)

  # Selecting classifier
  print(classifier)
  if classifier == 'MNB': # Multinomial Naive bayes
    clf = MultinomialNB(alpha=alpha)
  elif classifier == 'SVC': # Linear SVM
    clf = LinearSVC(random_state=127)
  elif classifier == 'LR': # Logistic regression
    clf = LogisticRegression()
  elif classifier == 'Random': # Random
    clf =DummyClassifier(strategy='uniform',random_state=100)
  elif classifier == 'GNB': # Gaussian naive bayes
    clf = GaussianNB()
  elif classifier == 'CNB': # Complement naive bayes
    clf = ComplementNB()
  elif classifier == 'ADAB': # Adaboost
    clf = AdaBoostClassifier(n_estimators=100)
  elif classifier == 'GRADB': # Gradient boosting
    clf = GradientBoostingClassifier(n_estimators=50)
  elif classifier == 'BAG': # Bagging with MultinomialNB
    clf = BaggingClassifier(base_estimator=LinearSVC(random_state=127), n_estimators=100, random_state=127)


  # Ensemble
  if classifier == 'STACK':
    estimators = [('lr', LogisticRegression()) , ('svc', LinearSVC(random_state=127))]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=127))

  if classifier == 'VOTE':
    clf = VotingClassifier(estimators=[('lr', LogisticRegression(random_state=127)), ('svc', SVC(kernel='linear',random_state=127))],voting='soft')

  # Training
  print('Training..')
  acc_score = clf.fit(X_train,y_train).score(X_val, y_val)
  print(f'Accuracy of {classifier} is', acc_score*100)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='/content/gdrive/MyDrive/NLP_project/trainset', help='Data path')
  parser.add_argument('--num_ex', type=int, default=1000, help='Number of examples')
  parser.add_argument('--classifier', type=str, default='SVC', help='Classifier')
  parser.add_argument('--alpha', type=float, default=1, help='Smoothing parameter for Multinomial Naive Bayes')
  args = parser.parse_args()
  train(args.data_path, args.num_ex, args.classifier, args.alpha)




