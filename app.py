# app.py
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from backend.indexer import index_site
from backend.retriever import HybridRetriever
from backend.embedder import embed_texts, get_openai_client
from backend.rag import build_prompt, generate_answer_with_openai

st.set_page_config(layout="wide")
st.title("Chat with Any Website — RAG Pipeline")

############################################
# Sidebar — Settings & Crawler
############################################
with st.sidebar:
    st.header("Website Crawler")

    url = st.text_input("Site URL", "")
    max_pages = st.number_input("Max Pages", min_value=1, max_value=500, value=20)

    use_llm = st.checkbox("Use OpenAI LLM for answers (requires OPENAI_API_KEY)", value=False)

    if st.button("Crawl & Build Index"):
        if not url.strip():
            st.error("Please enter a valid website URL.")
        else:
            with st.spinner("Crawling website and building index..."):
                try:
                    store, bm25_texts, metas = index_site(url, max_pages=max_pages)
                except Exception as e:
                    st.error(f"❌ Error indexing site: {e}")
                    st.stop()

                st.session_state["store"] = store
                st.session_state["bm25"] = bm25_texts
                st.session_state["metas"] = metas

            if not bm25_texts:
                st.warning("⚠ No readable content found. Try another site.")
            else:
                st.success("✅ Index created successfully!")


############################################
# Main Chat Area
############################################
query = st.text_input("Ask a question about the site:")

if st.button("Ask"):
    if "store" not in st.session_state:
        st.error("Please crawl a website first.")
        st.stop()

    store = st.session_state["store"]
    texts = st.session_state["bm25"]
    metas = st.session_state["metas"]

    if not texts or not metas:
        st.error("⚠ Index is empty; no text to search.")
        st.stop()

    # Embed query (local)
    qvec = embed_texts([query], provider="local")[0:1]

    retriever = HybridRetriever(store, texts, metas)
    try:
        retrieved = retriever.retrieve(query, qvec, top_k=5)
    except Exception as e:
        st.error(f"❌ Retrieval failed: {e}")
        st.stop()

    st.session_state["last_retrieved"] = retrieved

    # Conversation history
    history = st.session_state.get("history", [])

    # Build prompt for LLM
    prompt = build_prompt(query, retrieved, conversation_history=history)

    answer = None
    if use_llm:
        client = get_openai_client()
        if client is None:
            st.warning("OPENAI_API_KEY not set. Showing retrieved context only.")
        else:
            try:
                answer = generate_answer_with_openai(prompt)
            except Exception as e:
                st.error(f"❌ LLM error: {e}")

    # If no LLM answer (or LLM disabled), just show concatenated excerpt
    if answer is None:
        joined = "\n\n".join(item["meta"]["text"][:400] for item in retrieved)
        answer = "Here are the most relevant excerpts from the site:\n\n" + joined

    st.subheader("💬 Answer")
    st.write(answer)

    # Save conversation
    history.append({"role": "user", "text": query})
    history.append({"role": "assistant", "text": answer})
    st.session_state["history"] = history

############################################
# Right column — Sources
############################################
st.markdown("---")
st.subheader("📚 Sources")

retrieved = st.session_state.get("last_retrieved", [])
if not retrieved:
    st.caption("No sources yet. Ask a question after indexing a site.")
else:
    for i, item in enumerate(retrieved):
        meta = item["meta"]
        score = item["score"]
        title = meta.get("title") or "(no title)"
        url = meta.get("url") or ""
        text_preview = (meta.get("text") or "")[:600]

        st.markdown(f"**[{i}] {title}**  \nScore: `{score:.4f}`")
        if url:
            st.markdown(f"[Open source page]({url})")
        st.write(text_preview + ("..." if len(meta.get("text", "")) > 600 else ""))
        st.markdown("---")
