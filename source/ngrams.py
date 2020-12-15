from load import load_preprocessed_data
from nltk import ngrams

def compute_text_ngram(data, n):
    for subreddit in data:
        for i, post in enumerate(data[subreddit]):
            data[subreddit][i] = (post[0], list(ngrams(post[1], n)), post[2])
    return data

def main():
    train,dev,test = load_preprocessed_data()
    train_copy = train.copy()
    ngram_data = compute_text_ngram(train_copy, 6)
    #print first 5 posts of Advice subreddit
    print(ngram_data['Advice'][:5])

if __name__ == '__main__':
    main()