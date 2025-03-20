from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import json


class LLM:

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.5):
        self.model = model
        self.temperature = temperature


class User(BaseModel):
    name: str = Field(..., description="The name of the user")
    age: int = Field(..., description="The age of the user")
    # default_factory会创建一个LLM实例
    # llm: LLM = Field(default_factory=LLM, description="The LLM model to use")

    tool_choices: Literal["none", "auto", "required"] = "auto"

    class Config:
        arbitrary_types_allowed = True


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: Literal["system", "user", "assistant", "tool"] = Field(...)
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[str]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [
                tool_call.dict() for tool_call in self.tool_calls
            ]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        return message


if __name__ == "__main__":
    # user = User(name="John", age=20, tool_choices="auto")
    # print(user.model_json_schema())
    llm = LLM()
    print(llm)

    msg1 = Message(role="user", content="Hello, world!")
    msg2 = Message(role="assistant", content="Hello, too!")
    print(msg1 + msg2)

    print(msg1.model_dump())
    print(type(msg1.model_dump_json()))

    print(msg1.to_dict())
