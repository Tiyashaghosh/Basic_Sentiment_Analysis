from textblob import TextBlob
# from newspaper import Article

# url = 'https://en.wikipedia.org/wiki/COVID-19_pandemic'
# article = Article(url)
# article.download()
# article.parse()
# article.nlp()
# text = article.summary

with open('reviews.txt','r') as r:
    text = r.read()

blob = TextBlob(text)
sentiment = blob.sentiment.polarity # -1 to 1

print(sentiment)