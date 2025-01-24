from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, AsyncGenerator, Union, Optional
import json
import uuid

from utils.models import ReasoningModel
from utils.infer import Ollama, OpenAI
from utils.handle_tools import HandleTools
from services.database import DatabaseService

@dataclass
class MessageHandler:
    inference: Union[Ollama, OpenAI]
    handler: HandleTools
    db: DatabaseService

    def _create_json_message(self, message_type: str, **kwargs) -> str:
        base_message = {
            "message_type": message_type,
            **kwargs,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return json.dumps(base_message, indent=4)

    def create_error_message(self, tool_type: str, error: Exception) -> str:
        return self._create_json_message("error_message", 
                                        tool_type=tool_type, 
                                        content=str(error))
        
    async def process_response(self, response: ReasoningModel, session_id: uuid.UUID) -> str:
        try:
            return await self.handler.handle_tool(response, session_id)
        except Exception as e:
            return self.create_error_message(response.tool_args.tool_type, e)

    async def get_message_history(self, session_id: uuid.UUID) -> List[Dict[str, str]]:
        return await self.db.get_session_history(session_id)

    async def generate_response(self, session_id: uuid.UUID) -> ReasoningModel:
        message_history = await self.get_message_history(session_id)
        return await self.inference.generate_response(message_history)
    
    async def store_message(self, message: str, session_id: uuid.UUID, role: str, image_url: Optional[str]) -> None:
        await self.db.add_message(session_id, role, message, image_url)

    async def handle_message(self, message: str, session_id: uuid.UUID, image_url: Optional[str]) -> AsyncGenerator[dict, None]:
        message_json = self._create_json_message("user_message", content=message)
        await self.store_message(message_json, session_id, 'user', image_url)

        i = 0
        while True:
            if i > 10:
                break
            i += 1
            response = await self.generate_response(session_id)
            del response.reasoning
            
            await self.store_message(json.dumps(response.model_dump(), indent=4), session_id, 'assistant', None)

            processed_response = await self.process_response(response, session_id)
            if processed_response and isinstance(processed_response, str):
                await self.store_message(processed_response, session_id, 'user', None)
            
            if response.tool_args.tool_type == "send_message":
                yield {
                    "type": "text",
                    "content": response.tool_args.content
                }
                break
            else:
                yield {
                    "type": "status",
                    "content": f"Using tool: {response.tool_args.tool_type}..."
                }
                
async def main():
    handler = MessageHandler(Ollama(), HandleTools(), DatabaseService())
    async for message in handler.handle_message("Hello, world!", uuid.uuid4(), None):
        print(message)
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())