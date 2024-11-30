import os
import requests
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Configure NLTK data directory (optional but helps on Render)
nltk.data.path.append("/tmp/nltk_data")

# Download necessary NLTK data
nltk.download("punkt", download_dir="/tmp/nltk_data")
nltk.download("stopwords", download_dir="/tmp/nltk_data")

# Get the API key from environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

# Function to fetch news articles from NewsAPI
def fetch_news(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=us&category=technology&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        news_list = [
            {
                "Title": article["title"],
                "Description": article["description"],
                "URL": article["url"],
                "Published_At": article["publishedAt"],
            }
            for article in articles
        ]
        return pd.DataFrame(news_list)
    else:
        print(f"Failed to fetch news: {response.status_code}")
        return None

# Function to preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

# Main function
def main():
    print("Fetching news data...")
    df = fetch_news(API_KEY)
    if df is not None:
        print("Preprocessing news data...")
        df["Processed_Description"] = df["Description"].apply(preprocess_text)
        df.to_csv("/tmp/preprocessed_news_data.csv", index=False)
        print("Preprocessed data saved to '/tmp/preprocessed_news_data.csv'")
    else:
        print("No data to preprocess.")

if __name__ == "__main__":
    main()
