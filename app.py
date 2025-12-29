import os
import time
import requests
import pandas as pd
import streamlit as st

from dotenv import load_dotenv

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data():
	"""Ensure required NLTK datasets are available."""
	resources = [
		("sentiment/vader_lexicon", "vader_lexicon"),
		("tokenizers/punkt", "punkt"),
		("corpora/stopwords", "stopwords"),
		("corpora/wordnet", "wordnet"),
	]
	for path, pkg in resources:
		try:
			nltk.data.find(path)
		except LookupError:
			nltk.download(pkg, quiet=True)


def process_text(text: str) -> str:
	"""Tokenize, remove stopwords, lemmatize, and normalize text."""
	if not isinstance(text, str):
		text = ""

	tokens = word_tokenize(text.lower())
	english_stops = set(stopwords.words("english"))
	filtered_tokens = [t for t in tokens if t.isalpha() and t not in english_stops]

	lemmatizer = WordNetLemmatizer()
	lemmatized_tokens = [lemmatizer.lemmatize(t) for t in filtered_tokens]

	return " ".join(lemmatized_tokens)


def fetch_articles(api_key: str, topic: str, pages: int = 5) -> pd.DataFrame:
	"""Fetch NYTimes articles for a topic and return a DataFrame."""
	if not api_key:
		raise ValueError("Missing API_KEY in environment (.env)")

	headlines, snippets, abstracts, pub_dates,links = [], [], [], [],[]

	for i in range(pages):
		url = (
			f"https://api.nytimes.com/svc/search/v2/articlesearch.json"
			f"?q={requests.utils.quote(topic)}&api-key={api_key}&page={i}"
		)
		r = requests.get(url, timeout=20)
		if r.status_code != 200:
			raise RuntimeError(f"API error {r.status_code}: {r.text[:200]}")
		data = r.json()

		docs = (data.get("response") or {}).get("docs") or []
		for article in docs:
			headlines.append((article.get("headline") or {}).get("main") or "")
			links.append(article.get("web_url") or "")
			snippets.append(article.get("snippet") or "")
			abstracts.append(article.get("abstract") or "")
			pub_dates.append(article.get("pub_date") or None)

		# Be kind to the API
		time.sleep(0.5)

	df = pd.DataFrame(
		{
			"headline": headlines,
			"snippet": snippets,
			"abstract": abstracts,
			'link': links,
			"pub_date": pd.to_datetime(pub_dates, errors="coerce"),
		}
	)
	if not df.empty:
		df["pub_date"] = df["pub_date"].dt.date
	return df


def add_sentiments(df: pd.DataFrame) -> pd.DataFrame:
	"""Add sentiment columns and a commonSentiment majority label (0/1)."""
	if df.empty:
		return df

	analyzer = SentimentIntensityAnalyzer()

	def sentiment_label(text: str) -> int:
		scores = analyzer.polarity_scores(text or "")
		# Positive if compound is moderately positive; mirrors typical VADER threshold
		return 1 if scores.get("compound", 0.0) >= 0.05 else 0

	processed = df.copy()
	processed["headline_proc"] = processed["headline"].apply(process_text)
	processed["snippet_proc"] = processed["snippet"].apply(process_text)
	processed["abstract_proc"] = processed["abstract"].apply(process_text)

	processed["headlineSentiment"] = processed["headline_proc"].apply(sentiment_label)
	processed["snippetSentiment"] = processed["snippet_proc"].apply(sentiment_label)
	processed["abstractSentiment"] = processed["abstract_proc"].apply(sentiment_label)

	sums = (
		processed["headlineSentiment"]
		+ processed["snippetSentiment"]
		+ processed["abstractSentiment"]
	)
	processed["commonSentiment"] = (sums >= 2).astype(int)
	return processed


def main():
	nltk.download('all')
	# UI
	st.title("NYTimes Topic Sentiment Explorer")
	st.caption("Fetch NYTimes articles, process text, and summarize sentiment.")

	try:
		api_key = st.secrets["API_KEY"]
	except (KeyError, FileNotFoundError):
		from dotenv import load_dotenv
		load_dotenv("./.env")
		api_key = os.getenv("API_KEY")

	if not api_key:
		st.error("API_KEY not found. Add it to Streamlit secrets or .env file.")
		st.stop()

	# Inputs
	topic = st.text_input("Topic", value="AI")
	pages = st.number_input("Pages to fetch (NYT limit ~5/min)", min_value=1, max_value=5, value=3)

	fetch = st.button("Get Data")

	if fetch:
		try:
			with st.spinner("Preparing NLTK resources..."):
				ensure_nltk_data()

			with st.spinner("Fetching articles..."):
				df = fetch_articles(api_key, topic, pages)

			if df.empty:
				st.warning("No articles returned for this topic.")
				return

			with st.spinner("Analyzing sentiments..."):
				analyzed = add_sentiments(df)

			st.success(f"Fetched {len(analyzed)} articles.")

			# Sentiment counts instead of table
			pos_count = int((analyzed["commonSentiment"] == 1).sum())
			neg_count = int((analyzed["commonSentiment"] == 0).sum())
			st.subheader("Sentiment Summary")
			c1, c2 = st.columns(2)
			with c1:
				st.metric("Positive articles (1)", pos_count)
			with c2:
				st.metric("Negative articles (0)", neg_count)

			# Per-article cards
			st.subheader("Articles")
			for _, row in analyzed.iterrows():
				sentiment_text = "Positive" if int(row["commonSentiment"]) == 1 else "Negative"
				st.markdown(f"### {row['headline']}")
				st.write(row["abstract"])
				st.write(f"Published: {row['pub_date']}")
				st.link_button("Go to article", row["link"])
				st.info(f"Common Sentiment: {sentiment_text}")

		except Exception as e:
			st.error(f"Error: {e}")


if __name__ == "__main__":
	# Workaround Anaconda protobuf issue: prefer Python impl
	os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
	main()
