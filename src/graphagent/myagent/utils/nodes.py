from typing import Literal, Optional

class PlanningNode:
    """
    使用大模型进行任务规划的节点
    
    Args:
        api_key (str): 大模型api key，如果是本地部署的大模型例如vllm，需要设置api_key为"EMPTY"。
        base_url (str): 大模型base url。
        model (str, optional): 大模型名称，默认"qwen-plus"。
        tool_choice (str, optional): 工具选择模式，包括"auto", "required", "none"，默认"auto"。
        temperature (float, optional): 温度，控制大模型生成结果的随机性，越大随机性越强，默认0.7。
        max_tokens (int, optional): 最大tokens，默认1000。
        stream (bool, optional): 是否流式输出，默认False。
        enable_thinking (bool, optional): 是否启用大模型思考模式，仅在使用qwen3模型时有效，默认是None，表示该大模型不具备思考模式切换能力。
    """
    def __init__(self,
                 *,
                 api_key: str,
                 base_url: str,
                 model: str = "qwen-plus",
                 tool_choice: Literal["auto", "required", "none"] = "auto",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 stream: bool = False,
                 enable_thinking: Optional[bool] = None):
