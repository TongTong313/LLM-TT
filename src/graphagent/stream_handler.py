"""
流式输出处理器
用于捕获print输出并实时转发到WebSocket
"""

import asyncio
import json
from typing import Callable, Any
import sys
from io import StringIO
import logging


class StreamHandler:
    """流式输出处理器"""

    def __init__(self, websocket_send_func: Callable[[str], Any]):
        """
        初始化流式输出处理器
        
        Args:
            websocket_send_func: 发送消息到WebSocket的函数
        """
        self.websocket_send_func = websocket_send_func
        self.original_stdout = sys.stdout
        self.output_buffer = StringIO()
        self.captured_output = []

    def __enter__(self):
        """进入上下文管理器"""
        # 重定向stdout到我们的缓冲区
        sys.stdout = self.output_buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        # 恢复原始stdout
        sys.stdout = self.original_stdout

    async def process_output(self):
        """处理输出缓冲区中的内容"""
        content = self.output_buffer.getvalue()
        if content:
            # 获取新增的内容
            new_content = content[len(''.join(self.captured_output)):]
            if new_content:
                self.captured_output.append(new_content)
                # 发送到WebSocket
                await self.websocket_send_func(
                    json.dumps({
                        "type": "chunk",
                        "content": new_content,
                        "full_content": content
                    }))
            # 清空缓冲区
            self.output_buffer.truncate(0)
            self.output_buffer.seek(0)


class AsyncStreamHandler:
    """异步流式输出处理器"""

    def __init__(self, websocket_send_func: Callable[[str], Any]):
        """
        初始化异步流式输出处理器
        
        Args:
            websocket_send_func: 发送消息到WebSocket的函数
        """
        self.websocket_send_func = websocket_send_func
        self.original_print = print
        self.original_logger_handlers = {}
        self.captured_output = []

    def __enter__(self):
        """进入上下文管理器"""

        # 替换print函数
        def custom_print(*args, **kwargs):
            # 调用原始print
            self.original_print(*args, **kwargs)

            # 捕获输出
            output = " ".join(str(arg) for arg in args)
            if output.strip():
                self.captured_output.append(output)
                # 异步发送（注意：这里可能会有竞态条件）
                asyncio.create_task(self._send_output(output))

        # 替换全局print
        import builtins
        builtins.print = custom_print

        # 设置logger处理器
        self._setup_logger_handler()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        # 恢复原始print
        import builtins
        builtins.print = self.original_print

        # 恢复logger处理器
        self._restore_logger_handler()

    def _setup_logger_handler(self):
        """设置logger处理器"""

        # 创建自定义的logger处理器
        class WebSocketHandler(logging.Handler):

            def __init__(self, send_func):
                super().__init__()
                self.send_func = send_func

            def emit(self, record):
                try:
                    msg = self.format(record)
                    # 异步发送logger消息
                    asyncio.create_task(
                        self.send_func(
                            json.dumps({
                                "type": "log",
                                "level": record.levelname.lower(),
                                "content": msg,
                                "timestamp": record.created
                            })))
                except Exception:
                    pass

        # 为所有logger添加处理器
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            if not logger.handlers:  # 避免重复添加
                handler = WebSocketHandler(self.websocket_send_func)
                handler.setFormatter(logging.Formatter('%(message)s'))
                logger.addHandler(handler)
                self.original_logger_handlers[name] = handler

        # 为root logger也添加处理器
        if not logging.root.handlers:
            handler = WebSocketHandler(self.websocket_send_func)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logging.root.addHandler(handler)
            self.original_logger_handlers['root'] = handler

        # 处理loguru logger
        try:
            import loguru
            from loguru import logger as loguru_logger

            # 创建自定义的loguru sink
            class LoguruWebSocketSink:

                def __init__(self, send_func):
                    self.send_func = send_func

                def write(self, message):
                    try:
                        # 解析loguru消息格式，只提取消息内容
                        import re
                        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) \| (\w+) \| ([^:]+):([^:]+):(\d+) - (.+)'
                        match = re.match(pattern, message.strip())

                        if match:
                            timestamp, level, name, function, line, content = match.groups(
                            )
                            level = level.lower()

                            # 只保留"-"之后的内容，去掉时间戳和文件信息
                            content = content.strip()

                            # 过滤掉一些不必要的日志
                            if any(skip in content.lower() for skip in
                                   ['正在执行步骤', '推理过程', '调用大模型api失败']):
                                return  # 跳过这些日志

                            # 对于工具调用相关的日志，只显示关键信息
                            if 'tool' in function.lower(
                            ) or 'tool' in name.lower():
                                # 提取工具名称和参数
                                tool_pattern = r'调用工具[：:]\s*(\w+)\s*参数[：:]\s*(.+)'
                                tool_match = re.search(tool_pattern, content)
                                if tool_match:
                                    tool_name = tool_match.group(1)
                                    tool_params = tool_match.group(2)
                                    content = f"🔧 调用工具: {tool_name} | 参数: {tool_params}"
                                else:
                                    # 如果没有匹配到工具调用模式，检查是否是工具执行成功
                                    success_pattern = r'工具(\w+)执行成功'
                                    success_match = re.search(
                                        success_pattern, content)
                                    if success_match:
                                        tool_name = success_match.group(1)
                                        content = f"✅ 工具 {tool_name} 执行成功"
                                    else:
                                        # 其他工具相关日志，简化显示
                                        content = content.replace(
                                            'graphagent.node.tool:__call__:',
                                            '').strip()
                        else:
                            # 如果无法解析，使用默认格式
                            content = message.strip()
                            level = 'info'

                        # 异步发送loguru消息
                        asyncio.create_task(
                            self.send_func(
                                json.dumps({
                                    "type":
                                    "log",
                                    "level":
                                    level,
                                    "content":
                                    content,
                                    "source":
                                    "loguru",
                                    "timestamp":
                                    timestamp if match else None
                                })))
                    except Exception:
                        pass

                def flush(self):
                    pass

            # 添加loguru sink
            loguru_logger.add(
                LoguruWebSocketSink(self.websocket_send_func),
                format=
                "{time} | {level} | {name}:{function}:{line} - {message}",
                level="INFO")

            # 保存loguru sink引用以便后续移除
            self.loguru_sink = LoguruWebSocketSink(self.websocket_send_func)

        except ImportError:
            # loguru未安装，跳过
            self.loguru_sink = None
            pass

    def _restore_logger_handler(self):
        """恢复logger处理器"""
        for name, handler in self.original_logger_handlers.items():
            if name == 'root':
                logging.root.removeHandler(handler)
            else:
                logger = logging.getLogger(name)
                logger.removeHandler(handler)

        # 清理loguru sink
        if hasattr(self, 'loguru_sink') and self.loguru_sink is not None:
            try:
                import loguru
                from loguru import logger as loguru_logger
                # 移除loguru sink（通过重新配置logger）
                loguru_logger.remove()
                # 重新添加默认的console sink
                loguru_logger.add(
                    sys.stderr,
                    format=
                    "{time} | {level} | {name}:{function}:{line} - {message}")
            except ImportError:
                pass

    async def _send_output(self, output: str):
        """发送输出到WebSocket"""
        try:
            await self.websocket_send_func(
                json.dumps({
                    "type": "chunk",
                    "content": output,
                    "full_content": "".join(self.captured_output)
                }))
        except Exception as e:
            # 忽略发送错误
            pass


def create_stream_handler(websocket_send_func: Callable[[str], Any],
                          use_async: bool = False):
    """
    创建流式输出处理器
    
    Args:
        websocket_send_func: 发送消息到WebSocket的函数
        use_async: 是否使用异步处理器
        
    Returns:
        StreamHandler或AsyncStreamHandler实例
    """
    if use_async:
        return AsyncStreamHandler(websocket_send_func)
    else:
        return StreamHandler(websocket_send_func)
