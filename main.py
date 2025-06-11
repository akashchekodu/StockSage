from app.rag_qa import run_rag_pipeline
from dotenv import load_dotenv
import sys

load_dotenv()

print("📊 StockSage Financial Chatbot")
print("Type your question below (or type 'exit' to quit):\n")

while True:
    try:
        query = input("🧾 You: ").strip()
        if query.strip() is None:
            continue
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting StockSage. Stay informed!")
            sys.exit(0)
        
        print("🤖 Thinking...\n")
        answer, sources = run_rag_pipeline(query)

        print("📌 Answer:")
        print(answer)

        if sources:
            print("\n🔗 Sources:")
            for src in sources:
                print("-", src)
        print("\n" + "-"*60 + "\n")

    except KeyboardInterrupt:
        print("\n👋 Interrupted. Exiting StockSage.")
        break
