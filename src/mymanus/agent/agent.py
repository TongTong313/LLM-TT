# 智能体类，智能体包含感知、规划、工具、记忆等组件，一个一个来实现
from typing import List, Dict, Any, Optional, Callable
from .memory_manager import MemoryManager
from .tool_manager import ToolManager
import json
from openai import AsyncOpenAI
import os
from .llm import LLM


class Agent:
    """智能体类，由工具、记忆、规划、感知等模块构建，咱们一个一个来实现

    Version:
        v0.1 (2025-04-09):
            - 实现一个最简单的智能体
            - 智能体规划由一个简单大模型实现
            - 只包含工具模块和记忆模块
            - 具备React框架，先think，再act
            - 支持基本的对话功能
            - 支持工具调用

    Args:
        llm (LLM): 大模型实例，在这里主要用于任务规划
        tool_manager (ToolManager): 工具管理器
        memory_manager (MemoryManager): 记忆管理器
    """

    def __init__(self, llm: LLM, tool_manager: ToolManager,
                 memory_manager: MemoryManager):
        self.llm = llm
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager

    # 智能体支持对工具采用装饰器的形式变为注册工具
    def register_tool(self, func: Callable):
        """类似MCP，用装饰器直接注册工具"""

        def decorator(func: Callable):
            self.tool_manager.register_tool(func)
            return func

        return decorator
