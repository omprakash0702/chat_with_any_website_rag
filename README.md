# Chat with Any Website – RAG Pipeline

A web-based application that allows users to chat with the content of any website using:
- Web Scraping
- Text Chunking
- Vector Embeddings
- FAISS Vector Search
- Retrieval-Augmented Generation (RAG)
- Streamlit UI

## Features
- Crawl and index any public website
- Ask natural language questions about the site
- Hybrid retrieval (BM25 + vector search)
- Source-aware answers
- Works fully with local embeddings
- Optional OpenAI integration

## Tech Stack
- Python
- Streamlit
- FAISS
- Sentence Transformers
- BeautifulSoup
- Requests
- OpenAI (optional)

## Current Status
✅ Wikipedia & static sites supported  
🚧 Dynamic sites (Letterboxd, JS-heavy sites) – in progress  
🚀 Selenium-based rendering – coming next  

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

<img width="959" height="430" alt="image" src="https://github.com/user-attachments/assets/b13f66ca-fe0c-48ec-9ee6-40efa6d9b267" />




