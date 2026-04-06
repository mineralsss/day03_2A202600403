"""
CLI Chatbot — ReAct Agent with configurable LLM providers.
Run:  python chat.py
"""
import os
from dotenv import load_dotenv

load_dotenv()

from src.agent.agent import ReActAgent
from src.tools.search_product import TOOL_SPEC as search_spec
from src.tools.get_product_detail import TOOL_SPEC as detail_spec
from src.tools.check_inventory import TOOL_SPEC as inventory_spec
from src.tools.compare_product import TOOL_SPEC as compare_spec
from src.tools.web_search_product import TOOL_SPEC as web_search_spec
from src.tools.read_web_page import TOOL_SPEC as read_web_page_spec


def main():
    provider_type = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    model_name = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    
    if provider_type == "google" or provider_type == "gemini":
        from src.core.gemini_provider import GeminiProvider
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("❌ Error: Please set your GEMINI_API_KEY in the .env file.")
            return
        print("⏳ Initializing Gemini model...")
        llm = GeminiProvider(api_key=api_key)
        
    elif provider_type == "openai":
        from src.core.openai_provider import OpenAIProvider
        api_key = os.getenv("GITHUB_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not api_key or api_key in {"your_github_token_here", "your_openai_api_key_here"}:
            print("❌ Error: Please set GITHUB_TOKEN (or OPENAI_API_KEY) in the .env file.")
            return
        print("⏳ Initializing OpenAI-compatible model...")
        llm = OpenAIProvider(model_name=model_name, api_key=api_key)
        
    else:
        from src.core.local_provider import LocalProvider
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/gemma-4-E4B-it-Q4_1.gguf")
        print("⏳ Loading local model... (this may take a moment)")
        try:
            llm = LocalProvider(model_path=model_path)
        except ImportError as exc:
            print(f"❌ Error: {exc}")
            return
        
    print(f"✅ Model loaded: {llm.model_name}\n")

    tools = [search_spec, detail_spec, inventory_spec, compare_spec, web_search_spec, read_web_page_spec]
    agent = ReActAgent(llm, tools)

    print("=" * 50)
    print("  🤖  Shopping Chatbot  (type 'quit' to exit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye! 👋")
            break

        answer = agent.run(user_input)
        print(f"\n🤖 Bot: {answer}")


if __name__ == "__main__":
    main()
