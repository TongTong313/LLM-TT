from typing import List, Dict, Union
from pydantic import BaseModel, Field
# pydantic：将Python代码的数据类型验证实体化


class MemoryManager(BaseModel):
    """记忆管理器，用于存储对话历史
    
    Args:
        memory (List[Dict[str, str]]): 记忆
        max_memory (int): 最大记忆数
    """
    memory: List[Dict[str, str]] = Field(default_factory=list,
                                         description="记忆")
    max_memory: int = Field(default=10, description="最大记忆数")

    def add_message(self, message: Union[Dict[str, str], List[Dict[str,
                                                                   str]]]):
        """添加一条消息到记忆，超过最大记忆数则删除最早的消息

        Args:
            message (Dict[str, str]): 消息
        """
        if isinstance(message, Dict):
            self.memory.append(message)
        elif isinstance(message, List):
            self.memory.extend(message)
        else:
            raise ValueError("message must be a Dict or List")
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def get_memory(self) -> List[Dict[str, str]]:
        """获取记忆"""
        return self.memory

    def clear(self):
        """清空记忆"""
        self.memory = []


if __name__ == "__main__":
    MemoryManager(memory=[{
        "role": "system",
        "content": "你是一个AI助手，请根据用户的问题给出回答，可以采用工具调用帮助回答问题"
    }, {
        "role": "user",
        "content": "123"
    }],
                  max_memory=13.234)
