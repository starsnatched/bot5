from peewee import *

from typing import List, Dict, Optional
import logging
from datetime import datetime
import os
import uuid

db = SqliteDatabase(None)

class Message(Model):
    session_id = UUIDField()
    role = CharField()
    content = TextField()
    image_url = CharField(null=True) 
    timestamp = DateTimeField(default=datetime.now)

    class Meta:
        database = db
        
class DisabledTools(Model):
    tool_type = CharField()
    
    class Meta:
        database = db

class DatabaseService:
    def __init__(self, db_path: str = "./db/database.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_path()
        db.init(self.db_path)
        db.connect()
        db.create_tables([Message])
        
    def init_path(self):
        if not os.path.exists("./db"):
            os.makedirs("./db")

    async def init_db(self):
        if not Message.table_exists():
            db.create_tables([Message])
        if not DisabledTools.table_exists():
            db.create_tables([DisabledTools])
            
    async def get_disabled_tools(self) -> List[str]:
        try:
            tools = DisabledTools.select()
            return [tool.tool_type for tool in tools]
        except Exception as e:
            self.logger.error(f"Error retrieving disabled tools: {e}")
            raise
        
    async def remove_disabled_tool(self, tool_type: str) -> None:
        try:
            query = DisabledTools.delete().where(DisabledTools.tool_type == tool_type)
            query.execute()
        except Exception as e:
            self.logger.error(f"Error removing disabled tool from database: {e}")
            raise
        
    async def add_disabled_tool(self, tool_type: str) -> None:
        try:
            DisabledTools.create(tool_type=tool_type)
        except Exception as e:
            self.logger.error(f"Error adding disabled tool to database: {e}")
            raise
        
    async def add_message(self, session_id: uuid.UUID, role: str, content: str, image_url: Optional[str]) -> None:
        try:
            Message.create(
                session_id=session_id,
                role=role,
                content=content,
                image_url=image_url
            )
        except Exception as e:
            self.logger.error(f"Error adding message to database: {e}")
            raise

    async def get_session_history(self, session_id: uuid.UUID) -> List[Dict[str, str]]:
        try:
            messages = (Message
                    .select()
                    .where(Message.session_id == session_id)
                    .order_by(Message.timestamp))
            
            formatted_messages = []
            for msg in messages:
                if msg.image_url:
                    formatted_message = {
                        "role": msg.role,
                        "content": [
                            {"type": "text", "text": msg.content},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": msg.image_url,
                                },
                            },
                        ],
                    }
                else:
                    formatted_message = {
                        "role": msg.role,
                        "content": msg.content
                    }
                formatted_messages.append(formatted_message)
                
            return formatted_messages
        except Exception as e:
            self.logger.error(f"Error retrieving session history: {e}")
            raise
        
    def __del__(self):
        if not db.is_closed():
            db.close()