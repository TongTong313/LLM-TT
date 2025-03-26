# 智能体类，智能体包含感知、规划、工具、记忆等组件，一个一个来实现
from typing import List, Dict, Any, Optional
from tool_manager import ToolManager
import json


class Memory:
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


class Agent:
    """智能体类，负责与大模型交互并管理工具和记忆"""

    def __init__(self):
        self.tool_manager = ToolManager()
        self.memory = Memory()

    async def chat(self,
                   message: str,
                   model: str = "gpt-3.5-turbo-0613") -> str:
        """与大模型对话

        Args:
            message (str): 用户输入的消息
            model (str): 使用的模型名称

        Returns:
            str: 大模型的回复
        """
        # 将用户消息添加到历史记录
        self.memory.add_message("user", message)

        # 准备工具列表
        tools = []
        for tool_name, tool in self.tool_manager._tools.items():
            tools.append(tool.tool_schema)

        try:
            # 调用大模型（这里需要根据实际使用的API进行实现）
            response = await self._call_llm(messages=self.memory.get_history(),
                                            tools=tools,
                                            model=model)

            # 处理响应
            if self._has_function_call(response):
                # 执行函数调用
                result = await self._handle_function_call(response)
                # 将函数执行结果添加到历史记录
                self.memory.add_message("assistant", result)
                return result
            else:
                # 直接返回文本响应
                self.memory.add_message("assistant", response)
                return response

        except Exception as e:
            error_message = f"对话过程中发生错误：{str(e)}"
            self.memory.add_message("system", error_message)
            return error_message

    async def _call_llm(self, messages: List[Dict[str, str]],
                        tools: List[Dict], model: str) -> Dict:
        """调用大模型API

        Args:
            messages (List[Dict[str, str]]): 对话历史
            tools (List[Dict]): 可用工具列表
            model (str): 模型名称

        Returns:
            Dict: 大模型的响应
        """
        # 这里需要实现具体的API调用
        # 例如使用 OpenAI API：
        # response = await openai.ChatCompletion.create(
        #     model=model,
        #     messages=messages,
        #     tools=tools
        # )
        # return response
        pass

    def _has_function_call(self, response: Dict) -> bool:
        """检查响应中是否包含函数调用"""
        return "function_call" in response

    async def _handle_function_call(self, response: Dict) -> str:
        """处理函数调用

        Args:
            response (Dict): 大模型的响应

        Returns:
            str: 函数执行结果
        """
        try:
            function_call = response["function_call"]
            tool_name = function_call["name"]
            arguments = json.loads(function_call["arguments"])

            # 执行工具
            result = await self.tool_manager.execute_tool(
                tool_name, **arguments)
            return str(result)

        except Exception as e:
            return f"函数调用执行失败：{str(e)}"

    def register_tool(self, func, tool_name: Optional[str] = None):
        """注册工具"""
        self.tool_manager.register_tool(func, tool_name)

    def clear_memory(self):
        """清空对话历史"""
        self.memory.clear()
