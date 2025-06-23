from langchain_core.runnables.config import RunnableConfig


class BaseNode:
    """
    基础节点类，所有节点都需要继承该类
    """

    async def run(self, state, config: RunnableConfig):
        raise NotImplementedError("务必实现run方法，描述节点执行逻辑")

    async def __call__(self, state, config: RunnableConfig):
        """调用run方法，输入graph的state，输出修改后的state
        """
        state = await self.run(state, config)
        return state
