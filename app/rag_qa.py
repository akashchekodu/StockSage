from app.retriever import get_relevant_articles
from app.llm import call_mistral

def run_rag_pipeline(query: str):
    # 🔍 Step 1: Get dynamically relevant chunks
    documents = get_relevant_articles(query)

    if not documents:
        return (
            f"Sorry, I couldn’t find any recent or relevant news for: \"{query}\". "
            "This might be because there's no current coverage on this topic, or the query may need rephrasing for better results.", 
            []
        )


    # 🧱 Step 2: Build context without inline citations
    context_parts = []
    for doc in documents:
        cleaned_chunk = doc['chunk'].strip().replace("\n", " ")
        context_parts.append(cleaned_chunk)

    context = "\n\n".join(context_parts)

    # 🧠 Step 3: Construct the prompt
    prompt = f"""You are a financial assistant. Based only on the news below, answer the user's question with detailed explanation.
Use only the information provided.

News Articles:
{context}

Question: {query}
Answer:"""

    # 🤖 Step 4: Call the LLM
    answer = call_mistral(prompt)

    # 📎 Step 5: Collect unique sources
    unique_sources = list({(doc["source"], doc["link"]) for doc in documents})
    formatted_sources = [
        f"- [{source}]({link})" for source, link in unique_sources
    ]
    sources_section = "\n\nSources:\n" + "\n".join(formatted_sources)

    return answer.strip() + sources_section, formatted_sources
