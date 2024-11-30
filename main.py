import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import requests

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Get the API key from the environment variable
API_KEY = os.getenv("a4a17aabcc7f4c559f07be45d8704695")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

# Define the NewsAPI URL
url = f"https://newsapi.org/v2/top-headlines?country=us&category=technology&apiKey={API_KEY}"

# Fetch news data
response = requests.get(url)
if response.status_code == 200:
    articles = response.json().get('articles', [])
    news_list = [
        {
            'Title': article.get('title'),
            'Description': article.get('description'),
            'URL': article.get('url'),
            'Published_At': article.get('publishedAt')
        }
        for article in articles
    ]
    df = pd.DataFrame(news_list)
    print("News data fetched successfully!")
else:
    print(f"Failed to fetch news: {response.status_code}")
    exit()

# Preprocess text
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

df['Processed_Description'] = df['Description'].apply(preprocess_text)
print("Text preprocessing completed!")

# Save the processed data
df.to_csv("/tmp/preprocessed_news_data.csv", index=False)
print("Preprocessed data saved to '/tmp/preprocessed_news_data.csv'")
