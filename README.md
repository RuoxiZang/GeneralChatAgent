# 通用 Chat Agent

一个具备搜索能力和记忆管理的智能对话助手，基于 DeepInfra API 实现。

## 功能特性

- **🔍 搜索工具**：内置知识库检索，支持关键词匹配
- **🧠 记忆系统**：
  - 短期记忆：保留最近对话上下文
  - 长期记忆：自动提取并存储用户偏好信息
- **🤖 智能推理**：ReAct 模式，自主决定是否调用搜索工具

## 快速开始

### 1. 安装依赖

```bash
pip install openai python-dotenv
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
DEEPINFRA_API_KEY=your_actual_api_key_here
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
```

### 3. 运行

```bash
python agent.py
```

## 使用示例

```
User: 我叫小明,是一个Python程序员
[Memory] 提取到新记忆: 用户姓名：小明，职业：Python程序员
Agent: 你好，小明！很高兴认识你...

User: DeepInfra是做什么的？
[System] 触发搜索工具，查询关键词: DeepInfra
Agent: DeepInfra 提供高性价比的 LLM 推理 API 服务...

User: 结合我的职业，DeepInfra对我有什么用？
Agent: 作为 Python 程序员，DeepInfra 可以帮助你...
```

## 项目结构

```
GeneralChatAgent/
├── agent.py          # 主程序
├── .env              # 环境配置（需自行创建）
└── README.md         # 项目文档
```

## 核心组件

| 模块 | 说明 |
|------|------|
| `SearchTool` | 本地知识库搜索 |
| `MemoryManager` | 对话历史与用户画像管理 |
| `UniversalAgent` | Agent 核心逻辑 |

## 技术栈

- Python 3.8+
- OpenAI SDK
- DeepInfra API
- dotenv

## 注意事项

- 确保 `.env` 文件不被提交到版本控制（添加到 `.gitignore`）
- 默认使用 `Meta-Llama-3.1-70B-Instruct` 模型
- 短期记忆窗口默认保留最近 3 轮对话
