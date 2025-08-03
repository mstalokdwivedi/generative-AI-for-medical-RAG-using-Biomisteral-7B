# generative-AI-for-medical-RAG-using-Biomisteral-7B
import os
from langchain import LLMChain, PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load BioMistral‑Clinical (or standard BioMistral‑7B)
tokenizer = AutoTokenizer.from_pretrained("ZiweiChen/BioMistral-Clinical-7B")
model = AutoModelForCausalLM.from_pretrained("ZiweiChen/BioMistral-Clinical-7B")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, temperature=0.0)

llm = HuggingFacePipeline(pipeline=pipe)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load & chunk documents
loader = PyPDFLoader("medical_docs/sample.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# Build or load vectorstore
vectordb = Chroma.from_documents(chunks, embedder, persist_directory="chroma_db")
vectordb.persist()

# Prompt template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the medical question using only the following provided context.\n"
        "Do not hallucinate; if you cannot find the answer in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
)

qa_chain = LLMChain(llm=llm, prompt=prompt)

def answer_query(query: str):
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join(d.page_content for d in docs)
    resp = qa_chain.run(context=context, question=query)
    return resp

if __name__ == "__main__":
    while True:
        q = input("\nAsk medical question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("--- Answer ---")
        print(answer_query(q))
