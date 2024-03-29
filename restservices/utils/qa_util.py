
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import cohere

from utils import ingest_util
from app import data_models


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
co = cohere.Client(os.getenv("COHERE_API_KEY"))

pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "my-rag-index"
pinecone_index = pinecone.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


def get_recommendations(query, top_k=3, return_single="True"):
  single_response=True
  if (return_single=="True"):
      single_response = True
  else:
      single_response = False
  embed = ingest_util.get_embeddings([query])[0]
  #print(embed)
  reco = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True, include_values=False)

  metadata_arr=[]
  for x in reco['matches']:
      print("Metadata file name:" + x['metadata']['source'] + " page:" +
            str(x['metadata']['page']) + " score:" + str(x['score']))
      Metadata = data_models.Metadata()
      Metadata.sourceFileName=x['metadata']['source']
      Metadata.pageNo = int(x['metadata']['page'])
      Metadata.score = str(x['score'])
      metadata_arr.append(Metadata)

  contexts = [
      x['metadata']['text'] for x in reco['matches']
  ]
  print(len(contexts))
  # print(contexts)
  template = """Answer the question based on the context below.\n\n Context: {context} \n
      Question : {query} \n
      Answer: """
  llm = OpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0.0, api_key=OPENAI_API_KEY)
  prompt = PromptTemplate.from_template(template)

  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response_arr = []
  print("Response val:"+str(single_response))
  if (single_response):
    response_llm = llm_chain.run(context=contexts, query=query)
    response_arr.append(response_llm)
  else:
      for x in reco['matches']:
        # print("Metadata file name:"+x['metadata']['source']+" page:"+
        #       str(x['metadata']['page'])+" score:"+str(x['score']))
        response_llm = llm_chain.run(context=x['metadata']['text'] , query=query)
        response_arr.append(response_llm)
  print(response_arr)
  return response_arr,metadata_arr

def get_recommendations_withrerank(query, top_k=3):
  embed = ingest_util.get_embeddings([query])[0]

  reco = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True, include_values=False)
  docs = {x["metadata"]['text']: i for i, x in enumerate(reco["matches"])}
  metadata_arr = []
  for x in reco['matches']:
      print("Metadata file name:" + x['metadata']['source'] + " page:" +
            str(x['metadata']['page']) + " score:" + str(x['score']))
      Metadata = data_models.Metadata()
      Metadata.sourceFileName = x['metadata']['source']
      Metadata.pageNo = int(x['metadata']['page'])
      Metadata.score = str(x['score'])
      metadata_arr.append(Metadata)
  rerank_docs = co.rerank(
      query=query, documents=docs.keys(), top_n=25, model="rerank-english-v2.0"
  )
  contexts = [
      doc.document['text'] for doc in rerank_docs
  ]
  # print(contexts)
  template = """Answer the question based on the context below.\n\n Context: {context} \n
      Question : {query} \n
      Answer: """
  llm = OpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0.0, api_key=OPENAI_API_KEY)
  prompt = PromptTemplate.from_template(template)

  llm_chain = LLMChain(prompt=prompt, llm=llm)
  response_arr = []
  response_llm = llm_chain.run(context=contexts, query=query)
  response_arr.append(response_llm)
  return response_arr,metadata_arr
