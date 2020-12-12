I got the original data from https://files.pushshift.io/reddit/submissions/ in a file called RS_2020-04.zst. 

I used the python3 command line to separate out the data we're using here, so I don't have a source file to show for that

The data was moved into different .csv files with the data we'd need from the json data downloaded. We selected among the most popular subreddits in our RS_2020-04 dataset.
The header for the .csv's (Not included) is id,subreddit,title,selftext,num_comments,permalink

The chosen subreddits had the 10,000 posts with the highest num_comments chosen, which are contained in the data/csv directory per subreddit.

The data from these files was then tokenized, lemmatized and filtered for stopwords, and the title, selftext, and subreddit were output to files categorized by train/dev/test sets in data/json.
