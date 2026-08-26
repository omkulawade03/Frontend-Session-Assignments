import os
import sys
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline

# 1. Load environment variables
load_dotenv()


def initialize_chat_model():
    """Prompts the user to select execution mode and returns configured ChatHuggingFace model."""
    print("=" * 45)
    print("      DUAL-MODE CHATBOT INITIALIZATION       ")
    print("=" * 45)
    print("1. API Mode (deepseek-ai/DeepSeek-R1)")
    print("2. Local Mode (TinyLlama/TinyLlama-1.1B-Chat-v1.0)")

    choice = input("Select Mode (1 or 2): ").strip()

    if choice == "1":
        # Check API token validity
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            print("\nError: HUGGINGFACEHUB_API_TOKEN missing in environment or .env file.")
            sys.exit(1)

        try:
            print("\nInitializing API Model...")
            llm = HuggingFaceEndpoint(
                repo_id="deepseek-ai/DeepSeek-R1",
                task="text-generation",
                max_new_tokens=256,
                temperature=0.7,
                huggingfacehub_api_token=token,
            )
            return ChatHuggingFace(llm=llm)
        except Exception as e:
            print(f"\nFailed to load API model: {e}")
            sys.exit(1)

    elif choice == "2":
        try:
            print("\nLoading Local Model into memory (this may take a few moments)...")
            llm = HuggingFacePipeline.from_model_id(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",
                pipeline_kwargs={
                    "max_new_tokens": 150,
                    "do_sample": True,
                    "temperature": 0.7,
                },
            )
            return ChatHuggingFace(llm=llm)
        except Exception as e:
            print(f"\nFailed to load Local model: {e}")
            sys.exit(1)

    else:
        print("\nInvalid choice selected. Exiting.")
        sys.exit(1)


def main():
    # Initialize chosen chat model
    chat_model = initialize_chat_model()

    # 2. Define polite system prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a very polite, kind, and helpful AI assistant."),
        ("user", "{user_input}")
    ])

    # Build chain using Runnable syntax
    chain = prompt_template | chat_model

    print("\n" + "=" * 45)
    print("  Chatbot ready! Type 'exit' to quit.")
    print("=" * 45)

    # 3. Interactive continuous chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            # 4. Exit condition
            if user_input.lower() == "exit":
                print("\nAssistant: Goodbye! Have a wonderful day ahead!")
                break

            if not user_input:
                continue

            response = chain.invoke({"user_input": user_input})
            print(f"\nAI: {response.content}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")


if __name__ == "__main__":
    main()



    