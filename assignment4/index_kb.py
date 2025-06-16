import os
import json
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    model="text-embedding-ada-002",
    api_version="2023-05-15"
)

with open("self_critique_loop_dataset.json", "r", encoding="utf-8") as f:
    kb_data = json.load(f)

docs = []
for entry in kb_data:
    docs.append(Document(
        page_content=entry["answer_snippet"],
        metadata={
            "doc_id": entry["doc_id"],
            "source": entry["source"],
            "last_updated": entry["last_updated"],
            "question": entry["question"]
        }
    ))

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_kb")
print(f"Indexed {len(docs)} KB entries into FAISS (local file).")
