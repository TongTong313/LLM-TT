from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class OpenAIMessage(BaseModel):
    """通用OpenAI接口Message

    Args:
        role: 角色
        content: 内容
        tool_call_id: 工具调用ID，用于tool calling场景
        tool_calls: 工具调用，用于tool calling场景
    """
    role: str
    content: str = ''
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class UserMessage(OpenAIMessage):
    """用户消息，默认角色为user
    """
    role: str = 'user'


class AssistantMessage(OpenAIMessage):
    """助手消息，默认角色为assistant
    """
    role: str = 'assistant'


class ToolMessage(OpenAIMessage):
    """工具消息，默认角色为tool
    """
    role: str = 'tool'
    # 工具消息必须有tool_call_id
    tool_call_id: str


class SystemMessage(OpenAIMessage):
    """系统消息，默认角色为system
    """
    role: str = 'system'


if __name__ == '__main__':
    # 把pydantic转换为json格式
    us = UserMessage(content="Hello, world!")
    # 转换为Json格式
    print(us.model_dump_json())

    # 转换为字典格式
    print(us.model_dump())
