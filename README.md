# NYTimes Topic Sentiment Explorer (Streamlit)

This app fetches NYTimes articles for a chosen topic, processes text with NLTK, and computes a majority sentiment label across headline, snippet, and abstract.

## Setup

1. Create a `.env` file in the project root with your NYTimes API key:

```
API_KEY=YOUR_NYTIMES_API_KEY
```

Note: Avoid spaces around the variable name (e.g., use `NEWS_API` not `NEWS_API =`).

2. (Recommended) Use a virtual environment:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

-   If you're using Anaconda base environment and encounter protobuf errors, use the Python implementation:

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
streamlit run app.py
```

-   In a clean virtual environment:

```bash
streamlit run app.py
```

Open the URL shown (usually http://localhost:8501) and enter a topic, then click "Get Data" to fetch and analyze articles.
