from langchain_core.runnables.config import RunnableConfig


class BaseNode:
    """
    基础节点类，所有节点都需要继承该类
    """

    async def __call__(self, state, config: RunnableConfig):
        """必须要实现__call__方法，描述节点执行逻辑
        """
        raise NotImplementedError("务必实现__call__方法，描述节点执行逻辑")
