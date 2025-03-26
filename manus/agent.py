# 智能体类，智能体包含感知、规划、工具、记忆等组件，一个一个来实现
from tool_manager import ToolManager


class TFFAgent:

    def __init__(self):
        self.tool_manager = ToolManager()
