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
    prompt = f"""You are a helpful and precise financial assistant. Use only the information from the news articles below to answer the user's question. 
        Do not rely on prior knowledge or assumptions. If the answer is not present in the news, clearly state that.

        --- NEWS ARTICLES ---
        {context}

        --- USER QUESTION ---
        {query}

        --- INSTRUCTIONS ---
        - Base your answer strictly on the provided news articles.
        - Provide a clear and detailed explanation with references to the relevant article content.
        - If the information is incomplete or missing, say: "The available news does not provide enough information to answer this question reliably."

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
