import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

document_text = """Sick leave policy: Har employee ko saal mein 10 sick leaves milti hain. Sick leave lene ke liye manager ko ek din pehle inform karna zaroori hai.

Annual leave policy: Har employee 20 annual leaves le sakta hai. Annual leave 15 din pehle apply karni hoti hai.

Work from home policy: Employees hafte mein 2 din work from home kar sakte hain. Isay manager se pehle se approve karwana hota hai."""

# Document ko chunks mein todna (abhi ke liye simple tareeqa: double newline se)
chunks = document_text.split("\n\n")

chroma_client = chromadb.Client()
policy_collection = chroma_client.get_or_create_collection(name="company_policy")

result = client.models.embed_content(
    model="gemini-embedding-001",   # yeh sirf embeddings ke liye khaas model hai
    contents=chunks                  # poori list ek sath bhej dete hain
)
embeddings = [item.values for item in result.embeddings]
def setup_rag():
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    policy_collection.add(
        ids=ids,                # unique IDs
        documents=chunks,       # asal text
        embeddings=embeddings   # un ke numbers (vectors)
    )
setup_rag() 

def ask_rag(question: str):
    question_result = client.models.embed_content(
        model="gemini-embedding-001",   # yeh sirf embeddings ke liye khaas model hai
        contents=question               
    )
    question_embedding = question_result.embeddings[0].values

    results = policy_collection.query(
        query_embeddings=[question_embedding],   # list ke andar, kyunke ek se zyada sawal bhi pooch sakte hain
        n_results=1                               # sab se qareeb 1 chunk
    )
    retrieved_text = results["documents"][0][0]

    prompt = f"""Sirf neeche di gayi maloomat se jawab do.
    Agar jawab is maloomat mein nahi hai to kaho 'mujhe nahi pata'. 
    Maloomat:
    {retrieved_text}
    Sawal: {question}"""

    rag_response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

    return rag_response.text

