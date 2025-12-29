import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import requests
    from dotenv import load_dotenv
    import os
    import pandas as pd
    import time

    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    from scipy.stats import mode
    return (
        SentimentIntensityAnalyzer,
        WordNetLemmatizer,
        load_dotenv,
        mo,
        mode,
        os,
        pd,
        requests,
        stopwords,
        time,
        word_tokenize,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data collection""")
    return


@app.cell
def _(load_dotenv, os):
    # Load environment variables from .env file
    load_dotenv('./.env')

    # Get the API key
    api_key = os.getenv('API_KEY')
    return (api_key,)


@app.cell
def _(api_key, requests, time):
    topic = "AI"
    pages = 5 #NYtime API has a rate limit of 5 pages per minute

    # Create empty lists to store data
    snippets = []
    headlines = []
    abstract = []
    links = []
    pub_dates = []


    for i in range(pages):
        url = f"https://api.nytimes.com/svc/search/v2/articlesearch.json?q={topic}&api-key={api_key}&page={i}"
        r = requests.get(url)
        data = r.json()
        print(data)
        # Populate lists
        for article in data["response"]["docs"]:
            headlines.append(article["headline"]["main"])
            snippets.append(article["snippet"])
            abstract.append(article["abstract"])
            links.append(article["web_url"])
            pub_dates.append(article["pub_date"])

        time.sleep(0.5)
    return abstract, headlines, links, pub_dates, snippets


@app.cell
def _(abstract, headlines, links, pd, pub_dates, snippets):
    # Create DataFrame
    df = pd.DataFrame({
        'headline': headlines,
        'snippet': snippets,
        'abstract': abstract,
        'link': links,
        'pub_date': pd.to_datetime(pub_dates)
    })

    df["pub_date"] = df["pub_date"].dt.date
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Sentiment Analysis""")
    return


@app.cell
def _():
    #Only run this if it is the first time downloading nltk
    #nltk.download('all')
    return


@app.cell
def _(WordNetLemmatizer, df, stopwords, word_tokenize):
    processedDf = df.copy()

    def processText(text):
        tokens = word_tokenize(text.lower())

        filtered_tokens = [token for token in tokens if tokens not in stopwords.words('english')]

        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

        processed_text = ' '.join(lemmatized_tokens)

        return processed_text

    processedDf['headline'] = processedDf['headline'].apply(processText)
    processedDf['snippet'] = processedDf['snippet'].apply(processText)
    processedDf['abstract'] = processedDf['abstract'].apply(processText)
    processedDf
    return (processedDf,)


@app.cell
def _(SentimentIntensityAnalyzer, mode, processedDf):
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        scores = analyzer.polarity_scores(text)
        sentiment = 1 if scores['pos'] > 0 else 0
        return sentiment

    def row_mode(row):
        return mode(row.dropna(), keepdims=True)[0][0]


    processedDf['snippetSentiment'] = processedDf['snippet'].apply(get_sentiment)
    processedDf['headlineSentiment'] = processedDf['headline'].apply(get_sentiment)
    processedDf['abstractSentiment'] = processedDf['abstract'].apply(get_sentiment)
    processedDf['commonSentiment'] = processedDf[['snippetSentiment', 'headlineSentiment', 'abstractSentiment']].apply(row_mode, axis=1)
    processedDf
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
