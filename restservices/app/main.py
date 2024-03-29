import os
from fastapi import FastAPI, File, UploadFile
from typing import Union
import uvicorn

from app import data_models
from utils import qa_util, fileReader, ingest_util, validator

app = FastAPI(description="Query App", version="0.1.0")

@app.get("/")
def read_root():
    return {"msg": "Hello World"}


"""
This POST API accepts a Query and returns the response from a set of Private documents
The API takes optional parameters, top_k, modelName,temperature, top_p, max_tokens
"""
@app.post("/query",response_model=data_models.Output,summary="query a set of docs",description="This POST API accepts a query as a text \
The API takes optional parameters, modelName,top_k, temperature, top_p, max_tokens",response_description="The API returns the response of the Query, based on private documents \
", tags=["query"])
async def query_api(query:str, modelName: data_models.ModelName,
                    top_k_rag:Union[int,None]=3,rerank:Union[str,None] = "True", single_response:Union[str,None] = "True",temperature: Union[str, None] = "0.2",top_p: Union[str, None] = "0.4",max_tokens: Union[str, None] = "1024"):

    output = data_models.Output()
    output.status = "success"
    output.errorMsg = ""
    query_response=[]

    output = validator.validate(output, temperature, top_p, max_tokens)
    if (output.status=="failure"):
        return output
    print("top_k:"+str(top_k_rag))
    print("single_response:"+str(single_response))
    try:
        if (rerank=="True"):
            query_response, metadata_arr = qa_util.get_recommendations_withrerank(query, top_k_rag)
        else:
            query_response, metadata_arr = qa_util.get_recommendations(query, top_k_rag, return_single=single_response)

    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)


    output.response = query_response
    output.responseMetadataList = metadata_arr

    return output

"""
This POST API accepts a PDF file and returns status of ingestion
"""
@app.post("/ingest",response_model=data_models.OutputIngest,summary="ingest a PDF doc",description="This POST API accepts a PDF file\
", response_description="The API returns the status of ingestion", tags=["ingestdoc"])
async def ingest_api(file_data: UploadFile = File(...)):

    output = data_models.OutputIngest()
    output.status = "success"
    output.errorMsg = ""
    file_path="uploadDocs"
    filepath = os.path.join(file_path, file_data.filename)
    print(file_path)
    try:
        if not (os.path.exists(file_path)):
            os.mkdir(file_path)
        else:
            # Get the list of all files in the directory
            files = os.listdir(file_path)

            # Iterate over the list of files and delete each file
            for file in files:
                os.remove(os.path.join(file_path, file))
    except Exception as exc:
        print(
            f"Something went wrong in creating folder {file_path}. Error {exc}"
        )

    try:
        filepath = fileReader.read_doc(file_data, filepath)
        print(filepath)
        ingest_util.ingest_doc(filepath)
    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)


    return output

"""
This GET API returns the list of documents ingested
"""
@app.get("/listdocs",response_model=data_models.OutputRAG,summary="list the contents of RAG DB",description="This GET API lists the content of Vector DB", response_description="The API returns list of documents in VectorDB", tags=["listdocs"])
async def listdocs_api():

    output = data_models.OutputRAG()
    output.status = "success"
    output.errorMsg = ""

    try:
        listdocs = ingest_util.list_vectordb()
        print(listdocs)
        output.response = list(listdocs)
    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)

    return output

"""
This POST API deletes the document provided
"""
@app.post("/deletedocs",response_model=data_models.OutputRAG,summary="delete the specified document from RAG DB",description="This POST API deletes the document from Vector DB", response_description="The API returns success msg", tags=["deletedocs"])
async def deletedocs_api(fileName:str):

    output = data_models.OutputRAG()
    output.status = "success"
    output.errorMsg = ""

    try:
        id_arr = ingest_util.get_ids(fileName)
        ingest_util.delete_doc(id_arr)
    except Exception as error:
        output.status = "failure"
        output.errorMsg = "Failed to generate results:"+str(error)

    return output

if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False, root_path="/")