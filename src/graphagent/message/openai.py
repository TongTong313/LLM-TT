from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal


class OpenAIMessage(BaseModel):
    """通用OpenAI接口Message

    Args:
        role: 角色
        content: 内容
        tool_call_id: 工具调用ID，用于tool calling场景
        tool_calls: 工具调用，用于tool calling场景
    """
    role: Literal['user', 'assistant', 'tool', 'system']
    content: str = ''
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self):
        return self.model_dump()

    def to_json(self):
        return self.model_dump_json()

    @classmethod
    def user_message(cls, content: str) -> "OpenAIMessage":
        return cls(role='user', content=content)

    @classmethod
    def assistant_message(
            cls,
            content: str,
            tool_call_id: Optional[str] = None,
            tool_calls: Optional[List[Dict[str,
                                           Any]]] = None) -> "OpenAIMessage":
        return cls(role='assistant',
                   content=content,
                   tool_call_id=tool_call_id,
                   tool_calls=tool_calls)

    @classmethod
    def tool_message(cls, content: str, tool_call_id: str) -> "OpenAIMessage":
        return cls(role='tool', content=content, tool_call_id=tool_call_id)

    @classmethod
    def system_message(cls, content: str) -> "OpenAIMessage":
        return cls(role='system', content=content)


if __name__ == '__main__':
    # 把pydantic转换为json格式
    us = OpenAIMessage.user_message(content="Hello, world!")
    # 转换为Json格式
    print(us.model_dump_json())

    # 转换为字典格式
    print(us.model_dump())
