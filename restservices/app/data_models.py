from pydantic import BaseModel
from enum import auto
from fastapi_utils.enums import StrEnum

class ModelName(StrEnum):
    gpt_3_turbo_16k = auto()

class LLMModel(BaseModel):
    modelName:ModelName = ModelName.gpt_3_turbo_16k
    top_k:str="1"
    temperature:str="0.1"
    top_p:str="0.5"
    max_tokens:str="1024"


class Metadata(BaseModel):
    sourceFileName:str=""
    pageNo:int=0
    score:str=""


class Output(BaseModel):
    response:list[str]=[]
    responseMetadataList:list[Metadata]=[]
    status:str="success"
    errorCode:str="0"
    errorMsg:str=""

class OutputIngest(BaseModel):
    status:str="ingestion success"
    errorCode:str="0"
    errorMsg:str=""

class OutputRAG(BaseModel):
    response:list[str]=[]
    status:str="success"
    errorMsg:str=""


