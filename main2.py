import requests
from transformers import pipeline

API_KEY = open('API_KEY.txt').read().strip()
keyword = 'AAPL'
date = '2025-09-22'

pipe = pipeline("text-classification", model="ProsusAI/finbert")

url = (
    'https://newsapi.org/v2/everything?'
    f'q={keyword}&'
    f'from={date}&'
    'sortBy=popularity&'
    f'apiKey={API_KEY}'
)

response = requests.get(url)
response_json = response.json()

if response_json.get('status') != 'ok':
    print("Error fetching articles:", response_json.get('message'))
    articles = []
else:
    articles = response_json.get('articles', [])

filtered_articles = [
    article for article in articles
    if (keyword.lower() in article['title'].lower() or
        (article['description'] and keyword.lower() in article['description'].lower()))
]

total_score = 0
num_articles = 0

for article in filtered_articles:
    content = article['description'] or article.get('content', '')
    if not content:
        continue

    print(f"Title: {article['title']}")
    print(f"Link: {article['url']}")
    print(f"Published: {article['publishedAt']}")

    sentiment = pipe(content)[0]

    print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']}")
    print('-'*40)

    if sentiment['label'].lower() == 'positive':
        total_score += sentiment['score']
        num_articles += 1
    elif sentiment['label'].lower() == 'negative':
        total_score -= sentiment['score']
        num_articles += 1

if num_articles > 0:
    final_score = total_score / num_articles
    if final_score >= 0.15:
        overall = "Positive"
    elif final_score <= -0.15:
        overall = "Negative"
    else:
        overall = "Neutral"
    print(f"Overall Sentiment: {overall} ({final_score})")
else:
    print("No articles matched the keyword for sentiment analysis.")
