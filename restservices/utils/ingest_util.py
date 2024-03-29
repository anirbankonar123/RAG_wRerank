import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pinecone = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "my-rag-index"
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

INDEX_NAME = "my-rag-index"

if not(INDEX_NAME in [index.name for index in pinecone.list_indexes()]):
    pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
                          spec=PodSpec(environment='gcp-starter', pod_type="starter", pods=1))

index = pinecone.Index(INDEX_NAME)

def get_fileName(fileName):
    ind = fileName.rfind("/")
    if (ind>=0):
        fileName = fileName[ind + 1:]
    return fileName

def get_embeddings(articles):
   embedded_doc = embeddings.embed_documents(articles)
   # print(np.array(embedded_doc[0]).shape)
   return embedded_doc

def ingest_doc(fileName):
    CHUNK_SIZE = 1000
    loader = PyPDFLoader(fileName)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200
    )
    list_of_documents = []
    ctr=0
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

    if (len(prepped) >= 0):
        index.upsert(prepped)
    print("page ct:" + str(page_ctr))
    print("chunk ct:"+str(chunk_ctr))
    print("doc added to pinecone index")
    return

def list_vectordb():
    stats = index.describe_index_stats()
    print(stats)
    ret=[]
    res = index.query(vector=[0 for _ in range(1536)], top_k=10000,include_metadata=True, include_values=False)
    print(len(res['matches']))
    for x in res['matches']:
        ret.append(x['metadata']['source'])
    return set(ret)

def get_ids(fileName):

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

def delete_doc(id_arr):
    index.delete(
        ids = id_arr
    )