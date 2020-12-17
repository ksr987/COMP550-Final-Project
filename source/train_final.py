# Running this file returns the test accuracy of the best classifier 
# (Stacking classifier with Multinomial naive bayes and Complement naive bayes as base estimators and Linear svc as final estimator)

# NOTE : Install numpy, pandas, nltk, sklearn before running this script

# Importing libraries

import argparse
import numpy as np
import pandas as pd
import pickle
import sklearn
import nltk
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier , GradientBoostingClassifier , StackingClassifier , VotingClassifier

def train(data_path : str, num_ex : int, classifier : str, alpha : float, ngrams : tuple):
  """
  Loads data and modififies data to bag of words which is input to train and test the model

  :param data_path: Path to unigram data file
  :param num_ex: Number of examples per each class(subreddit)
  :param classifier: Multinomial NaiveBayes or LinearSVM or LogisticRegression or Complement Naive bayes or Ensemble classifiers
  :param alpha: Smoothing parameter for Multinomial naive bayes
  :param ngrams: ngram features to be extracted
  
  """

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
  vect = TfidfVectorizer(ngram_range=ngrams)

  # Creating bag of words with features (unigram/ bigram/ .. ngrams)
  data = pd.DataFrame(vect.fit_transform(list_of_reviews).toarray(), index=list_of_reviews, columns= vect.get_feature_names()) # Creating bag of words with unigram features
  data.index = train_data.index
  # Adding label 
  data['label_or_Class'] = train_data['title']

  # Shuffling
  data = data.sample(frac=1, random_state=27)
  X = data.iloc[:,:-1] # X - bag of words (input)
  y = data.iloc[:,-1]  # y - labels

  # Splitting train, test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=127)

  # Selecting classifier
  print(classifier)
  if classifier == 'MNB': # Multinomial Naive bayes
    clf = MultinomialNB(alpha=alpha)
  elif classifier == 'SVC': # Linear SVM
    clf = LinearSVC(random_state=127)
  elif classifier == 'LR': # Logistic regression
    clf = LogisticRegression()
  elif classifier == 'CNB': # Complement naive bayes
    clf = ComplementNB()
  elif classifier == 'BAG': # Bagging with Complement naive bayes
    clf = BaggingClassifier(base_estimator=ComplementNB(), n_estimators=100, random_state=127)


  # Ensemble
  if classifier == 'STACK':
    estimators = [('mnb', MultinomialNB(alpha=alpha)) , ('cnb', ComplementNB())]
    clf = StackingClassifier(estimators=estimators, final_estimator=LinearSVC(random_state=127))

  # Training
  print('Training..')
  clf.fit(X_train,y_train)
  pred_test = clf.predict(X_test)
  acc_score = accuracy_score(y_test, pred_test)
  pred_train = clf.predict(X_train)
  train_score = accuracy_score(y_train, pred_train)
  print('CLassifier - Stacking classifier with with Multinomial naive bayes \
  and Complement naive bayes as base estimators and Linear svc as final estimator')
  print(f'Train Accuracy is', train_score*100) 
  print(f'Test Accuracy is', acc_score*100)





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='../data/unigram', help='Data path')
  parser.add_argument('--num_ex', type=int, default=1000, help='Number of examples per each class(subreddit)')
  parser.add_argument('--classifier', type=str, default='STACK', help='Classifier')
  parser.add_argument('--alpha', type=float, default=1, help='Smoothing parameter for Multinomial Naive Bayes')
  parser.add_argument('--ngrams', type=int, nargs="+", help='ngrams to be extracted')
  args = parser.parse_args()
  if args.ngrams is None:
    args.ngrams = (1, 1)
  else:
    args.ngrams = tuple(args.ngrams)
  print("ngrams:", args.ngrams)
  train(args.data_path, args.num_ex, args.classifier, args.alpha, args.ngrams)

  # To select ngrams parameter run this way

  # !python train.py --data_path 'Path to input file' --ngrams 1 1 --num_ex 1000

  # 1 1 extracts only unigrams, 1 2 extracts uni+bigrams, 2 2 extracts only bigrams and so on




