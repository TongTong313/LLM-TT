"""
LLM-TT Web应用配置示例
请根据你的实际情况修改这些配置
"""

import os
from typing import Optional


class Config:
    """应用配置类"""

    # API配置
    OPENAI_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 模型配置
    DEFAULT_MODEL: str = "qwen-plus"  # 或其他支持的模型
    MAX_TOKENS: int = 8000
    TEMPERATURE: float = 0.7

    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # 工具配置
    ENABLE_SEARCH: bool = True
    ENABLE_CALCULATOR: bool = True
    ENABLE_TIME: bool = True

    # 安全配置
    CORS_ORIGINS: list = ["*"]  # 生产环境中应该限制具体域名
    RATE_LIMIT: Optional[int] = None  # 每分钟请求限制

    @classmethod
    def get_runnable_config(cls):
        """获取运行时配置"""
        return {
            'configurable': {
                'planning_max_tokens': cls.MAX_TOKENS,
                'planning_model': cls.DEFAULT_MODEL,
                'planning_temperature': cls.TEMPERATURE,
                'router_max_tokens': cls.MAX_TOKENS,
                'router_model': cls.DEFAULT_MODEL,
                'router_temperature': cls.TEMPERATURE,
            }
        }

    @classmethod
    def validate(cls):
        """验证配置"""
        if cls.OPENAI_API_KEY == "your-api-key-here":
            print("⚠️  警告: 请设置正确的OPENAI_API_KEY")
            return False
        return True


# 使用示例
if __name__ == "__main__":
    # 检查配置
    if Config.validate():
        print("✅ 配置验证通过")
        print(f"模型: {Config.DEFAULT_MODEL}")
        print(f"最大token: {Config.MAX_TOKENS}")
        print(f"温度: {Config.TEMPERATURE}")
    else:
        print("❌ 配置验证失败")
        print("请设置环境变量或修改配置")
