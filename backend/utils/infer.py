from openai import AsyncOpenAI
from ollama import AsyncClient
from txtai.pipeline import TextToSpeech

import chromadb
from chromadb.config import Settings

import uuid
import torch
from typing import List, Dict
from decouple import config
from datetime import datetime

from utils.models import ReasoningModel
from utils.get_config import get_config as gc

class OpenAI:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=config('OPENAI_API_KEY'))
        self.chroma_client = chromadb.PersistentClient(path="./db", settings=Settings(anonymized_telemetry=False))
        self.collection = self.chroma_client.get_or_create_collection(name="memory")
        self.tts = TextToSpeech("NeuML/kokoro-fp16-onnx")
        
    async def generate_voice(self, text: str) -> torch.Tensor:
        audio, _ = await self.tts(text, speaker="af_bella")
        return audio
        
    async def retrieve_memory(self, query: str, session_id: uuid.UUID) -> str:
        response = await self.client.embeddings.create(
            model=gc('openai.embedding_model'),
            input=query
        )
        results = self.collection.query(
            query_embeddings=[response.data[0].embedding],
            n_results=1,
            where={"session_id": session_id}
        )
        if results['documents'][0] == []:
            return "Memory not found."
        
        return results['documents'][0][0]
    
    async def store_memory(self, memory: str, session_id: uuid.UUID) -> str:
        memory = memory + "\nTIMESTAMP: " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        response = await self.client.embeddings.create(
            model=gc('openai.embedding_model'),
            input=memory
        )
        self.collection.add(
            ids=[uuid.uuid4().hex],
            embeddings=[response.data[0].embedding],
            documents=[memory],
            metadatas=[{"session_id": session_id}]
        )
        
        return "Memory stored successfully."
        
    async def generate_response(self, messages: List[Dict[str, str]]) -> ReasoningModel:
        response = await self.client.beta.chat.completions.parse(
            model=gc('openai.model'),
            messages=messages,
            response_format=ReasoningModel
        )
        
        return response.choices[0].message.parsed
    
class Ollama:
    def __init__(self) -> None:
        self.client = AsyncClient(host=gc('ollama.host'))
        self.chroma_client = chromadb.PersistentClient(path="./db", settings=Settings(anonymized_telemetry=False))
        self.collection = self.chroma_client.get_or_create_collection(name="memory")
        self.tts = TextToSpeech("NeuML/kokoro-fp16-onnx")
        
    async def generate_voice(self, text: str) -> torch.Tensor:
        audio, _ = await self.tts(text, speaker="af_bella")
        return audio
        
    async def retrieve_memory(self, query: str, session_id: uuid.UUID) -> str:
        response = await self.client.embeddings(
            model=gc('ollama.embedding_model'),
            prompt=query
        )
        results = self.collection.query(
            query_embeddings=[response.embedding],
            n_results=1,
            where={"session_id": session_id}
        )
        if results['documents'][0] == []:
            return "Memory not found."
        
        return results['documents'][0][0]
        
    async def store_memory(self, memory: str, session_id: uuid.UUID) -> str:
        memory = memory + "\nTIMESTAMP: " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        response = await self.client.embeddings(
            model=gc('ollama.embedding_model'),
            prompt=memory
        )
        self.collection.add(
            ids=[uuid.uuid4().hex],
            embeddings=[response.embedding],
            documents=[memory],
            metadatas=[{"session_id": session_id}]
        )
        
        return "Memory stored successfully."
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> ReasoningModel:
        response = await self.client.chat(
            model=gc('ollama.model'),
            messages=messages,
            format=ReasoningModel.model_json_schema(),
            options={
                "num_ctx": gc('ollama.num_ctx'),
            }
        )

        return ReasoningModel.model_validate_json(response.message.content)