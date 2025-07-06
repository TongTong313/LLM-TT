#!/usr/bin/env python3
"""
LLM-TT 智能对话机器人 - 增强版
支持流式输出、logger捕获、特殊效果和Markdown渲染
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """主函数"""
    print("🤖 LLM-TT 智能对话机器人 - 增强版")
    print("=" * 50)
    print("✨ 新功能:")
    print("   📋 Logger信息捕获")
    print("   🎨 特殊效果显示")
    print("   📝 Markdown渲染")
    print("   🌊 流式输出优化")
    print("=" * 50)

    # 检查API密钥
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("⚠️  警告: 未设置DASHSCOPE_API_KEY环境变量")
        print("请设置环境变量:")
        print("export DASHSCOPE_API_KEY='your-api-key'")
        print("或")
        print("set DASHSCOPE_API_KEY=your-api-key  # Windows")
        response = input("是否继续启动? (y/N): ")
        if response.lower() != 'y':
            return

    print("\n🚀 启动增强版Web服务器...")
    print("📝 服务器将在 http://localhost:8000 启动")
    print("💡 按 Ctrl+C 停止服务器")
    print("-" * 50)

    try:
        # 使用uvicorn直接启动
        subprocess.run([
            sys.executable, "-m", "uvicorn", "src.graphagent.web_server:app",
            "--host", "0.0.0.0", "--port", "8000", "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请确保已安装依赖: pip install fastapi uvicorn")


if __name__ == "__main__":
    main()
