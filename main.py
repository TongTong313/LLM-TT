from mymanus.prompt import SYSTEM_PROMPT as system_prompt
from mymanus.agent import BaseAgent, ToolManager, MemoryManager, LLM
from mymanus.tool import *
import os
from loguru import logger
import asyncio
import traceback

MAX_STEP = 5


async def main():
    # 初始化大模型
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 初始化大模型->智能体的大脑
    llm = LLM(api_key=api_key,
              base_url=base_url,
              model="qwen-plus-latest",
              max_tokens=8000,
              tool_choice="auto",
              stream=True,
              enable_thinking=True)

    # 初始化工具管理器
    tool_manager = ToolManager()
    # 初始化记忆管理器
    memory_manager = MemoryManager(max_memory=20)
    # 初始化智能体
    agent = BaseAgent(llm=llm,
                      tool_manager=tool_manager,
                      memory_manager=memory_manager,
                      max_step=MAX_STEP)

    # 注册工具
    agent.add_tool(baidu_search, tool_name="baidu_search")
    agent.add_tool(get_current_time, tool_name="get_current_time")
    agent.add_tool(terminate, tool_name="terminate")
    agent.add_tool(add, tool_name="add")

    while True:
        try:
            prompt_list = [{"role": "system", "content": system_prompt}]
            prompt = input(
                "我是童发发开发的manus超级助手，请输入你的需求，我会尽力解决你的问题，输入quit/exit可退出：")
            if prompt.lower() in ["quit", "exit"]:
                logger.warning("再见!")
                break

            # 要把prompt变为字典送入
            prompt_dict = [{"role": "user", "content": prompt}]
            prompt_list.extend(prompt_dict)

            # 运行智能体
            logger.warning(f"智能体正在运行中……")
            result = await agent.run(prompt_list)

            if result:
                logger.warning(f"智能体执行完成")
        except KeyboardInterrupt:
            logger.warning("再见!")
            break
        except Exception as e:
            logger.error(f"智能体运行错误: {e}")
            logger.error("错误堆栈信息:")
            logger.error(traceback.format_exc())
            break


if __name__ == "__main__":
    asyncio.run(main())
