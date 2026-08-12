from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

prompt = "Write a short creative story about a robot learning to cook."
temperatures = [0.1, 0.7, 1.2]

for temp in temperatures:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temp)
    response = llm.invoke(prompt)
    print(f"--- Temperature: {temp} ---")
    print(response.content)
    print("\n" + "="*40 + "\n")


    