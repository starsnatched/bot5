from utils.models import ReasoningModel
from utils.infer import Ollama, OpenAI
from utils.get_config import get_config as gc

import json
from typing import Any, Union, Tuple
    
class HandleTools:
    def __init__(self) -> None:
        self.get_inference()
        
    def get_inference(self) -> None:
        if gc("backend.llm_backend") == "openai":
            self.inference = OpenAI()
        elif gc("backend.llm_backend") == "ollama":
            self.inference = Ollama()
        else:
            raise ValueError("Invalid inference type.")
        
    @staticmethod
    def create_tool_return_json(tool_type: str, content: Any) -> str:
        return json.dumps({
            "message_type": "tool_return",
            "tool_type": tool_type,
            "content": content
        }, indent=4)
        
    async def handle_tool(self, tool_args: ReasoningModel, session_id: str) -> Union[str, Tuple[bytes, int]]:
        if tool_args.tool_args.tool_type == "send_message":
            return
        
        if tool_args.tool_args.tool_type == "store_memory":
            await self.inference.store_memory(tool_args.tool_args.memory, session_id)
            return self.create_tool_return_json(tool_args.tool_args.tool_type, "Memory stored successfully.")
        
        if tool_args.tool_args.tool_type == "retrieve_memory":
            memory = await self.inference.retrieve_memory(tool_args.tool_args.query, session_id)
            return self.create_tool_return_json(tool_args.tool_args.tool_type, memory)
        
        return self.create_tool_return_json(tool_args.tool_args.tool_type, "Tool does not exist.")