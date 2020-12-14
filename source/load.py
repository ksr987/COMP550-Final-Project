import json

subreddits = ['Advice','AskReddit','Coronavirus','TheTwitterFeed','gaming','gardening','modernwarfare','politics','relationship_advice','techsupport']

def load_preprocessed_data():
    train,dev,test = {},{},{}

    for subreddit in subreddits:
        train_file = open(f'../data/json/preprocessed_{subreddit}_lemmatized_and_filtered_stopwords_train.json')
        train[subreddit] = [(data['title'],data['text'],data['class']) for line in train_file if (data := json.loads(line))]
        train_file.close()
        dev_file = open(f'../data/json/preprocessed_{subreddit}_lemmatized_and_filtered_stopwords_dev.json')
        dev[subreddit] = [(data['title'],data['text'],data['class']) for line in dev_file if (data := json.loads(line))]
        dev_file.close()
        test_file = open(f'../data/json/preprocessed_{subreddit}_lemmatized_and_filtered_stopwords_test.json')
        test[subreddit] = [(data['title'],data['text'],data['class']) for line in test_file if (data := json.loads(line))]
        test_file.close()
    return train,dev,test


def main():
    train,dev,test = load_preprocessed_data()

    print('Testing loader:\n')
    for subreddit in subreddits:
        print(subreddit)
        print(f'\tTrain: has {len(train[subreddit])}')
        print(f'\tDev: has {len(dev[subreddit])}')
        print(f'\tTest: has {len(test[subreddit])}')

if __name__ == '__main__':
    main()

    
