# 童发发的大模型学习之旅

作者：Tong Tong

B站首页：[Double童发发](https://space.bilibili.com/323109608)

License：MIT License，请严格遵守项目协议规范，否则保留追究责任的权利。

## 1. 项目简介
童发发学习大模型的过程记录，会不断更新，包含代码、部分jupyter notebook讲稿等。

## 2. 项目结构
```
- README.md 说明文档
- bilibili 存在B站视频对应的jupyter讲稿
  - rope.ipynb 旋转位置编码讲稿
  - func_calling.ipynb 函数调用讲稿
- src 源代码
  - mcp 测试MCP协议
  - mymanus 从零开始构建manus！
- ref_code 参考代码
- fc_test.py 测试大模型function calling
- main.py 测试mymanus
```

## 3. 如何运行代码？

1. 安装uv：uv是一个更加先进的python包管理工具，安装方法见[uv官网](https://docs.astral.sh/uv/getting-started/installation/)

2. 从github上git clone本项目

```bash
git clone https://github.com/TongTong313/LLM-TT.git
```

3. 在根目录下安装所有依赖，有了uv后，安装依赖就非常简单了，你甚至可以直接跳过这一步到最后一步运行代码，它会自动给你装好~

```bash
uv sync
```

4. 为了保证包的引用可以不涉及相对路径啥的，依托pyproject.toml文件，把我们这个项目作为package安装一下

```bash
uv pip install -e .
```

5. 运行代码，这一步会自动安装所有依赖，**虽然如此但你不能跳过第4步**，这里就举一个例子，不同的项目当然要运行不同的代码哈哈~

```bash
# 运行manus代码，等价于python main.py
uv run main.py  
```

6. 如果遇到不能运行的情况，用下面的命令切换python环境到这个文件夹下的`.venv`，反正就是找到`.venv`文件夹，然后运行它里面的一个`activate`文件，这个文件在不同系统下可能会不一样~

```bash
# for linux/macos
source .venv/bin/activate 
# for windows
.\.venv\Scripts\activate 
```

## 4. 更新日志
- 2025.05.21
  - 部分类改用pydantic的BaseModel
  - python版本更新到3.13
- 2025.05.06
  - 采用uv管理项目依赖
  - 支持qwen3模型的enable_thinking功能
- 2025.05.01
  - 对工具调用bug做了一些修复
  - 函数调用支持List类型
  - 增加项目协议规范
- 2025.04.22
  - 计划增加MCP协议的支持
- 2025.03.19
  - 更新README.md
  - 增加`manus`文件夹，计划从零开始构建`manus`智能体
  - 增加`MCP`文件夹，测试MCP协议
- 2025.02.25
  - 更新README.md
  - 在`bilibili`文件夹中增加旋转位置编码jupyter讲稿`rope.ipynb`
  - 公开本项目