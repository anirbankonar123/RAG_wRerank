This repo provides a batch program to test RAG pipeline using Pinecone Free Acct, OpenAI and Cohere<br>

Before running this, the following is needed:<br>

Create a free Pinecone Acct in PineCone Console app.pinecone.io, and get your Pinecone Key<br>
Create a OpenAI acct in Openai console and get your Open AI key<br>
Create a Cohere Account in Cohere console dashboard.cohere.com and get your Cohere key<br>
Set these in env variable<br>

export PINECONE_API_KEY=<br>
export OPENAI_API_KEY=<br>
export COHERE_API_KEY=<br>

Install the following dependencies:<br>

openai==0.28.1<br>
langchain==0.0.346<br>
langchain_community<br>
pyPDF<br>
fastapi<br>
python-multipart<br>
cohere<br>
fastapi_utils<br>
pinecone<br>

To run the batch code: python testRAG.py<br>
In the main method, uncomment the method needed<br>
ex: to ingest doc, process_doc<br>
to list documents, list_vectordb<br>
to delete documents, get_ids, delete_doc<br>

To create the new Pinecone instance , turn the create_DB=True<br>

To run REST API: cd restservices<br>
uvicorn app.main:app --reload<br>
Go to localhost:8000/docs to test using Swagger interface<br>
