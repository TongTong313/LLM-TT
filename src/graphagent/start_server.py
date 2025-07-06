#!/usr/bin/env python3
"""
启动聊天机器人服务器
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def main():
    """主函数"""
    try:
        print("正在启动聊天机器人服务器...")
        from web_server import app
        import uvicorn

        print("✓ 服务器模块加载成功")
        print("启动地址: http://localhost:8000")
        print("WebSocket地址: ws://localhost:8000/ws/chat")
        print("按 Ctrl+C 停止服务器")

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所需依赖:")
        print("pip install fastapi uvicorn")
        return 1
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
