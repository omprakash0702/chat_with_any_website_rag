# backend/rag.py
from backend.embedder import get_openai_client

def build_prompt(query: str, retrieved_chunks: list[dict], conversation_history=None) -> str:
    """
    retrieved_chunks: list of {"meta": {...}, "score": float}
    """
    history_text = ""
    if conversation_history:
        for turn in conversation_history[-6:]:
            history_text += f"{turn['role'].upper()}: {turn['text']}\n"

    context_blocks = []
    for i, item in enumerate(retrieved_chunks):
        meta = item["meta"]
        title = meta.get("title") or "(no title)"
        url = meta.get("url") or ""
        text = meta.get("text") or ""
        context_blocks.append(f"[{i}] {title} ({url})\n{text}")

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are helping the user **understand the specific page or small set of pages** they gave.

FOCUS RULES:
- Focus on the **main subject of the page(s)**, not on navigation menus, portals, or site-wide categories.
- If the content looks like a **film list** (like Letterboxd), explain:
  - what the list is measuring,
  - how it is ranked,
  - and give some example films from near the top.
- If the user just says things like "Summarize" / "Summanrize", give a clear summary of:
  - what this page is about,
  - key points,
  - and any notable examples mentioned.
- Ignore generic links to other lists, categories, or unrelated pages.

CONTEXT:
{context}

CONVERSATION:
{history_text}

QUESTION:
{query}

Answer concisely, but with enough detail to be useful. When you use information from a block, cite it like [0], [1], etc.
"""
    return prompt.strip()


def generate_answer_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not set or openai client unavailable.")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
