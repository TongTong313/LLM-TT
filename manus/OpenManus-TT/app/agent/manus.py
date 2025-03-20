from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.google_search import GoogleSearch
from app.tool.python_execute import PythonExecute


class Manus(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends PlanningAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools")

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # Add general-purpose tools to the tool collection
    # 这里包含了五个工具（重写了ToolCallAgent的available_tools）：
    # 1. PythonExecute()：执行Python代码
    # 2. GoogleSearch()：执行Google搜索
    # 3. BrowserUseTool()：执行浏览器操作
    # 4. FileSaver()：保存文件
    # 5. Terminate()：终止程序
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(PythonExecute(), GoogleSearch(
        ), BrowserUseTool(), FileSaver(), Terminate()))

    max_steps: int = 20
