import chromadb

class RagService():
    def __init__(self,client):
        self.client = client
        self.chroma_client = chromadb.Client()
        self.policy_collection = self.chroma_client.get_or_create_collection(name="company_policy")
    
    def setup_rag(self):
        document_text = """Sick leave policy: Every employee gets 10 sick leaves per year. To take sick leave, the employee must inform their manager one day in advance.
Annual leave policy: Every employee can take 20 annual leaves. Annual leave must be applied for 15 days in advance.
Work from home policy: Employees can work from home 2 days per week. This must be approved by the manager in advance."""
        chunks = [c.strip() for c in document_text.split("\n")]
        result = self.client.models.embed_content(
            model="gemini-embedding-001",  
            contents=chunks                 
        )
        embeddings = [item.values for item in result.embeddings]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.policy_collection.add(
            ids=ids,               
            documents=chunks,       
            embeddings=embeddings   
        )

    def ask_rag(self, question: str):
        prompt = f"""Answer only using the information below.
If the answer is not in this information, say "I don't know".
Information:
{retrieved_text}
Question: {question}"""

        rag_response = self.client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
        return rag_response.text

    def retrieve(self, question: str):
        question_result = self.client.models.embed_content(
            model="gemini-embedding-001",  
            contents=question               
        )
        question_embedding = question_result.embeddings[0].values

        results = self.policy_collection.query(
            query_embeddings=[question_embedding],  
            n_results=1                             
        )
        retrieved_text = results["documents"][0][0]
        return retrieved_text
