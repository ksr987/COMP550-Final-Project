import csv

subreddits = ['Advice','AskReddit','Coronavirus','TheTwitterFeed','gaming','gardening','modernwarfare','politics','relationship_advice','techsupport']

class Data:
    def __init__(self, line):
        self.id = line[0]
        self.subreddit = line[1]
        self.title = line[2]
        self.selftext = line[3]
        self.num_comments = line[4]
        self.permalink = line[5]
        self.total = line
    def __str__(self):
        return f'Id:{self.id},Subreddit:{self.subreddit},Title:{self.title},Selftext:{self.selftext},Num Comments:{self.num_comments},Permalink:{self.permalink}'

def load_data():
    return {subreddit : [Data(line) for line in csv.reader(open('../data/' + subreddit + '.csv','r',newline=''))] for subreddit in subreddits}

def main():
    data = load_data()

    for subreddit in subreddits:
        print(data[subreddit][0])


if __name__ == '__main__':
    main()
