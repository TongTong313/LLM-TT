from typing import List, Dict


class MemoryManager:
    """记忆管理器，用于存储对话历史
    
    Args:
        max_memory (int): 最大记忆数
    """

    def __init__(self, max_memory: int = 10):
        self.memory: List[Dict[str, str]] = []
        self.max_memory = max_memory

    def add_message(self, message: Dict[str, str]):
        """添加一条消息到记忆，超过最大记忆数则删除最早的消息

        Args:
            message (Dict[str, str]): 消息
        """
        self.memory.append(message)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def get_memory(self) -> List[Dict[str, str]]:
        """获取记忆"""
        return self.memory

    def clear(self):
        """清空记忆"""
        self.memory.clear()
