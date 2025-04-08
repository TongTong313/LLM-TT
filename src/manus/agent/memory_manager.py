from typing import List, Dict


class MemoryManager:
    """记忆管理器，用于存储对话历史"""
    def __init__(self, max_history: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_history = max_history

    def add_message(self, role: str, content: str):
        """添加一条消息到历史记录"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history(self) -> List[Dict[str, str]]:
        """获取历史记录"""
        return self.history

    def clear(self):
        """清空历史记录"""
        self.history = []
