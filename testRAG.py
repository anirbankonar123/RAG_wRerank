
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec
from DLAIUtils import Utils
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import cohere

import langchain
print(langchain.__version__)
import openai
print(openai.__version__)


#Ref : https://www.pinecone.io/learn/series/rag/rerankers/

# init client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
create_DB=False

utils = Utils()
PINECONE_API_KEY = utils.get_pinecone_api_key()
OPENAI_API_KEY = utils.get_openai_api_key()
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "my-rag-index"
if (create_DB):
    if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
      pinecone.delete_index(INDEX_NAME)
      print("DB deleted")

    pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
      spec=PodSpec(environment='gcp-starter', pod_type="starter", pods = 1))
    print("new DB created")

index = pinecone.Index(INDEX_NAME)
def get_embeddings(articles):
   embedded_doc = embeddings.embed_documents(articles)
   # print(np.array(embedded_doc[0]).shape)
   return embedded_doc

def list_vectordb(index):
    stats = index.describe_index_stats()
    print(stats)
    ret=[]
    res = index.query(vector=[0 for _ in range(1536)], top_k=10000,include_metadata=True, include_values=False)
    print(len(res['matches']))
    for x in res['matches']:
        ret.append(x['metadata']['source'])
    return set(ret)

def get_ids(fileName,index):

    ret=[]
    #res = index.query(vector=[0 for _ in range(1536)], top_k=10000,include_metadata=True, include_values=False)
    res = index.query(
        vector=[0 for _ in range(1536)],
        top_k=10000,
        filter={
            "source": {"$eq": fileName},
        },
        include_metadata=True,
        include_values=False
    )

    print(len(res['matches']))
    id = []
    for x in res['matches']:
        id.append(x['id'])
        ret.append(x['metadata']['source'])
    print(set(ret))
    return id

def get_fileName(fileName):
    ind = fileName.rfind("/")
    fileName = fileName[ind + 1:]
    return fileName

def delete_doc(id_arr):
    index.delete(
        ids = id_arr
    )

def process_doc(fileName,index):
    CHUNK_SIZE = 1000
    loader = PyPDFLoader(fileName)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )

    page_ctr=0
    prepped = []
    chunk_ctr=0
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        embeddings_vector = get_embeddings(chunks)

        chunk_num = 0
        for chunk in chunks:
            fileName = get_fileName(fileName)
            id = fileName+"#"+str(chunk_ctr)
            prepped.append({'id': id, 'values': embeddings_vector[chunk_num],
                        'metadata': {'source': fileName,'page':page_ctr,'text':chunk}  } )
            chunk_num += 1
            chunk_ctr += 1

        if (len(prepped)>=20):
            index.upsert(prepped)
            prepped=[]

        page_ctr+=1

    if (len(prepped) > 0):
        index.upsert(prepped)

    print("page ct:" + str(page_ctr))
    print("chunk ct:"+str(chunk_ctr))
    print("doc added to pinecone index")
    return

def get_docs(index, query: str, top_k: int):
    # encode query
    xq = get_embeddings([query])[0]
    # search pinecone index
    res = index.query(xq, top_k=top_k, include_metadata=True)
    # get doc text
    #print(res["matches"])
    docs = {x["metadata"]['text']: i for i, x in enumerate(res["matches"])}
    return docs



def get_recommendations(pinecone_index, query, top_k=3, return_single=True):
  embed = get_embeddings([query])[0]
  #print(embed)
  reco = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True, include_values=False)

  for x in reco['matches']:
      print("Metadata file name:" + x['metadata']['source'] + " page:" +
            str(x['metadata']['page']) + " score:" + str(x['score']))
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

  if (return_single):
      print(llm_chain.run(context=contexts, query=query))
  else:
      for x in reco['matches']:
        print("Metadata file name:"+x['metadata']['source']+" page:"+
              str(x['metadata']['page'])+" score:"+str(x['score']))
        print(llm_chain.run(context=x['metadata']['text'] , query=query))
  return

def get_recommendations_withrerank(pinecone_index, query, top_k=3):
  embed = get_embeddings([query])[0]
  #print(embed)
  reco = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True, include_values=False)
  docs = {x["metadata"]['text']: i for i, x in enumerate(reco["matches"])}

  for x in reco['matches']:
      print("Metadata file name:" + x['metadata']['source'] + " page:" +
            str(x['metadata']['page']) + " score:" + str(x['score']))
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

  print(llm_chain.run(context=contexts, query=query))
  return

if __name__ == "__main__":
    print("hello")


    # articles_index = pinecone.Index(INDEX_NAME)
    # print(list_vectordb(articles_index))
    #id_arr = get_ids("Invoice-32646.pdf",articles_index)
    # # print(len(id_arr))
    #delete_doc(id_arr)

    # #process_doc("/home/anish/Downloads/IPCC_AR6_SYR_SPM.pdf",articles_index)
    # #process_doc("/home/anish/Downloads/Invoice-32646.pdf",articles_index)
    #
    # query = 'What book did Anirban buy and from whom?'
    # reco = get_recommendations(articles_index, query , top_k=1, return_single=True)
    #
    # reco = get_recommendations_withrerank(articles_index, query, top_k=5)

    # docs = get_docs(articles_index,query,5)
    #
    # print("\n---\n".join(docs.keys()))
    #
    # rerank_docs = co.rerank(
    #     query=query, documents=docs.keys(), top_n=25, model="rerank-english-v2.0"
    # )
    # print("after RERANK")
    # #print(rerank_docs)
    # for doc in rerank_docs:
    #     print("\n---\n")
    #     print(doc.document['text'])
    # print([docs[doc.document["text"]] for doc in rerank_docs])

    # res = list_vectordb(articles_index)
    # print(res)






