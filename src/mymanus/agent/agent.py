from typing import List, Dict, Optional, Callable
import json
from .memory_manager import MemoryManager
from .tool_manager import ToolManager
from .llm import LLM
from loguru import logger
from ..prompt import NEXT_STEP_PROMPT, FINAL_STEP_PROMPT
from pydantic import BaseModel, Field


class BaseAgent(BaseModel):
    """智能体基类

    Args:
        llm (LLM): 大模型实例
        tool_manager (ToolManager): 工具管理器
        memory_manager (MemoryManager): 记忆管理器
    """

    llm: LLM = Field(..., description="大模型实例")
    tool_manager: ToolManager = Field(..., description="工具管理器")
    memory_manager: MemoryManager = Field(..., description="记忆管理器")

    class Config:
        # 为什么要加这个配置？
        # 因为LLM、ToolManager、MemoryManager都不是pydantic能自动校验的类型，讲白了不是python自带的而是你自己定义的类，所以要加这个配置，否则报错！
        arbitrary_types_allowed = True

    async def run(self, message: List[Dict]):
        """运行智能体

        Args:
            message (List[Dict]): 用户的一句话query
        """
        raise NotImplementedError("子类必须实现run方法")


class ToolCallingAgent(BaseAgent):
    """ToolCallingAgent，由工具、记忆、规划、感知等模块构建，咱们一个一个来实现

    ToolCallingAgent特点：
        - 一个最简单的智能体
        - 智能体规划由一个简单大模型实现
        - 只包含工具模块和记忆模块
        - 具备React框架，先think，再act
        - 支持基本的对话功能
        - 支持工具调用
        - 后续的智能体可以继承这个基座智能体，并在此基础上添加更多的功能

    Args:
        llm (LLM): 大模型实例，在这里主要用于任务规划（继承自BaseAgent）
        tool_manager (ToolManager): 工具管理器（继承自BaseAgent）
        memory_manager (MemoryManager): 记忆管理器（继承自BaseAgent）
        max_step (int): 最大步骤，默认10
    """
    max_step: int = Field(default=10, description="最大步骤")
    next_step_prompt: str = Field(default=NEXT_STEP_PROMPT,
                                  description="下一步提示")
    final_step_prompt: str = Field(default=FINAL_STEP_PROMPT,
                                   description="最后一步提示")

    # React框架，先think（reasoning），再act
    async def think(self, message: List[Dict]) -> bool:
        """使用大模型进行思考，返回是否需要使用工具
        
        Args:
            message (List[Dict]): 消息列表

        Returns:
            bool: 是否需要使用工具
        """
        # 添加终止提示
        message.append({"role": "user", "content": self.next_step_prompt})
        response = await self.llm.chat(
            messages=message, tools=self.tool_manager.get_tool_schema_list())

        # 回复内容全部加入记忆模块，加入的得是字典
        self.memory_manager.add_message(response.model_dump())
        # 打印回复内容，流式输出会自动打印，不必要重复打印
        if response.content and not self.llm.stream:
            logger.info(f"智能体回复：{response.content}")
        # 判断是否需要使用工具
        if response.tool_calls:
            return True
        else:
            return False

    async def act(self, message: List[Dict]) -> bool:
        """调用对应工具返回结果，并将返回结构通过assistant message返回，需要注意，一旦调用工具还需要反馈assistant message，要把函数运行结果返回给大模型做下一步的计划
        
        Args:
            message (List[Dict]): 消息列表

        Returns:
            bool: 是否执行完所有的工具
        """

        # 根据记忆读取最新的回复，根据tool_calls顺序执行工具，返回的可能不止一个工具
        for tool_call in message["tool_calls"]:

            # 拿到调用工具的名称、入参、id、index
            tool_name = tool_call["function"]["name"]
            tool_arguments = tool_call["function"]["arguments"]
            tool_id = tool_call["id"]

            # 执行工具
            logger.info(f"调用工具：{tool_name}，入参：{tool_arguments}")
            try:
                # 如果tool_arguments为空字典，则不传入参数
                if tool_arguments == "{}":
                    tool_result = await self.tool_manager.execute_tool(
                        tool_name)
                else:
                    # 将tool_arguments转换为字典
                    tool_arguments = json.loads(tool_arguments)
                    tool_result = await self.tool_manager.execute_tool(
                        tool_name, **tool_arguments)
                logger.info(f"工具{tool_name}执行成功")

                # 然后是一个tool message
                tool_message = {
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_id,
                }
                self.memory_manager.add_message(tool_message)

                if tool_call["function"]["name"] == "terminate":
                    logger.warning(f"智能体认为任务完成，终止工具调用")
                    return True

            except Exception as e:
                logger.error(f"工具{tool_name}执行失败，错误信息：{e}")
                # 将错误信息告知大模型
                assistant_message = {
                    "content": f"工具{tool_name}执行失败，考虑调用其他工具",
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                }
                self.memory_manager.add_message(assistant_message)
        # 返回结果
        return False

    async def run_step(self, message: List[Dict]):
        """运行一个react步骤，包括一次think和一次act
        
        Args:
            message (List[Dict]): 消息列表

        Returns:
            bool: 是否是最后一步，达到终止条件
        """

        # 思考
        logger.warning(f"智能体正在思考……")
        should_act = await self.think(message)
        if should_act:
            # 行动
            # 获取最新的message
            logger.warning(f"智能体正在行动……")
            current_message = self.memory_manager.get_memory()[-1]
            should_terminate = await self.act(current_message)
            if should_terminate:
                return True
            else:
                return False
        else:
            return False

    async def run(self, message: List[Dict]):
        """运行完整轮数的react过程

        Args:
            message (List[Dict]): 用户的一句话query
        """

        # 用户问题本身也要加到记忆里面
        self.memory_manager.add_message(message)
        step = 0
        while step < self.max_step:
            logger.warning(f"正在执行第{step+1}步……")
            # 输入全量的message
            final_step = await self.run_step(self.memory_manager.get_memory())
            if final_step:
                break
            step += 1

        # 最后一步要综合除了最后一轮信息给用户一个总结性的回复，还需要和大模型做一次对话
        if final_step:

            final_message = {"role": "user", "content": self.final_step_prompt}
            # 注意在调用terminate工具的同时还可能有输出，得把terminate当成一个普通工具对待
            # 把final_message加入到memory当中
            self.memory_manager.add_message(final_message)

            logger.warning(f"智能体正在总结答案……")
            # 这里有一个特别坑的地方，就是tools必须全程保持一致，否则大模型自动进入新的问答，无法结合上下文信息分析了
            final_response = await self.llm.chat(
                messages=self.memory_manager.get_memory(),
                tool_choice="none",
                tools=self.tool_manager.get_tool_schema_list())
            self.memory_manager.add_message(final_response.model_dump())
            # 空一行
            print()
            logger.warning(f"智能体总结答案完成~")

        if step == self.max_step:
            logger.warning(f"智能体执行已达最大步数{self.max_step}")

        logger.warning("童发发的Manus超级助手已帮你解决当前问题，有其他问题还可问我哦~")
        logger.warning(f"智能体执行完成，记忆清空~")
        self.memory_manager.clear()

    # 智能体支持对工具采用装饰器的形式变为注册工具
    def tool(self, func: Callable, tool_name: Optional[str] = None):
        """类似MCP协议，用装饰器直接注册工具
        
        Args:
            func (Callable): 要注册的工具函数
            tool_name (Optional[str]): 工具名称，如果为None，则使用函数名作为工具名称
        """

        def decorator(func: Callable):
            self.add_tool(func, tool_name)
            return func

        return decorator

    def add_tool(self,
                 func: Callable,
                 tool_name: Optional[str] = None) -> None:
        self.tool_manager.register_tool(func, tool_name)
