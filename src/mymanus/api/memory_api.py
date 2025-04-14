from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
from pathlib import Path

app = FastAPI()


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    created_at: str
    updated_at: str


class MemoryManager:

    def __init__(self, storage_dir: str = "conversations"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def _get_conversation_path(self, conversation_id: str) -> Path:
        return self.storage_dir / f"{conversation_id}.json"

    def create_conversation(self, conversation_id: str) -> Conversation:
        """创建新的对话"""
        now = datetime.now().isoformat()
        conversation = Conversation(conversation_id=conversation_id,
                                    messages=[],
                                    created_at=now,
                                    updated_at=now)
        self._save_conversation(conversation)
        return conversation

    def add_message(self, conversation_id: str,
                    message: Message) -> Conversation:
        """添加消息到对话"""
        conversation = self._load_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404,
                                detail="Conversation not found")

        message.timestamp = datetime.now().isoformat()
        conversation.messages.append(message)
        conversation.updated_at = datetime.now().isoformat()

        self._save_conversation(conversation)
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取对话"""
        return self._load_conversation(conversation_id)

    def get_all_conversations(self) -> List[str]:
        """获取所有对话ID"""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def _load_conversation(self,
                           conversation_id: str) -> Optional[Conversation]:
        """从文件加载对话"""
        path = self._get_conversation_path(conversation_id)
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return Conversation(**data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load conversation: {str(e)}")

    def _save_conversation(self, conversation: Conversation):
        """保存对话到文件"""
        path = self._get_conversation_path(conversation.conversation_id)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(conversation.dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save conversation: {str(e)}")


# 创建内存管理器实例
memory_manager = MemoryManager()


@app.post("/conversations/{conversation_id}")
async def create_conversation(conversation_id: str):
    """创建新的对话"""
    return memory_manager.create_conversation(conversation_id)


@app.post("/conversations/{conversation_id}/messages")
async def add_message(conversation_id: str, message: Message):
    """添加消息到对话"""
    return memory_manager.add_message(conversation_id, message)


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """获取对话"""
    conversation = memory_manager.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@app.get("/conversations")
async def get_all_conversations():
    """获取所有对话ID"""
    return memory_manager.get_all_conversations()
