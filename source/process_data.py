import csv
import json
import random
from sys import stdout
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

stops = stopwords.words('english')
trans_table = str.maketrans('()[],-.?!:;#&','             ')
subreddits = ['Advice','AskReddit','Coronavirus','TheTwitterFeed','gaming','gardening','modernwarfare','politics','relationship_advice','techsupport']
lemmatizer = WordNetLemmatizer()

class CSVData:
    def __init__(self, line):
        self.id = line[0]
        self.subreddit = line[1]
        self.title = line[2]
        self.selftext = line[3]
        self.num_comments = line[4]
        self.permalink = line[5]
    def __str__(self):
        return f'Id:{self.id},Subreddit:{self.subreddit},Title:{self.title},Selftext:{self.selftext},Num Comments:{self.num_comments},Permalink:{self.permalink}'

def load_data():
    return {subreddit : [CSVData(line) for line in csv.reader(open('../data/csv/' + subreddit + '.csv','r',newline=''))] for subreddit in subreddits}

def unigrams(line,pos_tags=False):
    if pos_tags:
        return pos_tag(list(filter(lambda str: str.isalnum(),word_tokenize(line.lower()))))
    else:
        return list(filter(lambda str: str.isalnum(),list(word_tokenize(line.lower()))))

def lemmatize(unigram_list): #Requires that output from unigrams(..) has pos_tags=True
    def lemmatize_on_list(unigram_list):
        return [lemmatizer.lemmatize(word, tag[0].lower()) if tag[0].lower() in ['a', 'r', 'n', 'v'] else word for (word,tag) in unigram_list]
    return lemmatize_on_list(unigram_list)

#def stem(unigram_lists):#Requires that output from unigrams(...) has pos_tags=False
#    stemmer = LancasterStemmer()
#    def stem_list(unigram_list):
#        return [stemmer.stem(unigram) for unigram in unigram_list]
#    return [stem_list(unigram_list) for unigram_list in unigram_lists]

def process_list_to_file(_list,_file, total):
    def process_item_to_file(item,_file):
        data = {}
        data['title'] = list(filter(lambda x: x not in stops, lemmatize(unigrams(item.title.lower().translate(trans_table),pos_tags=True)))) 
        data['text'] = list(filter(lambda x: x not in stops, lemmatize(unigrams(item.selftext.lower().translate(trans_table),pos_tags=True))))
        data['class'] = item.subreddit.lower()
        #print(json.dumps(data))
        _file.write(json.dumps(data) + '\n')
    count = 0
    for item in _list:
        process_item_to_file(item,_file)
        count += 1
        stdout.write(f'\t\t\tProgress: {count/total * 100:.2f}%          \r')
        stdout.flush()
    stdout.write('                                          \r')


def main():
    data = load_data()
    print('Creating preprocessed files')
    for subreddit in subreddits:
        random.shuffle(data[subreddit])
        print(f'\tStarting {subreddit}')
        print('\t\tStarting train file')
        train_file = open(f'../data/json/preprocessed_{subreddit}_lemmatized_and_filtered_stopwords_train.json','w')
        process_list_to_file(data[subreddit][:6000],train_file,6000)
        print('\t\tStarting dev file')
        dev_file = open(f'../data/json/preprocessed_{subreddit}_lemmatized_and_filtered_stopwords_dev.json','w')
        process_list_to_file(data[subreddit][6000:8000],dev_file,2000)
        print('\t\tStarting test file')
        test_file = open(f'../data/json/preprocessed_{subreddit}_lemmatized_and_filtered_stopwords_test.json','w')
        process_list_to_file(data[subreddit][8000:], test_file,2000)
    print('Finished preprocessed files')

if __name__ == '__main__':
    main()
