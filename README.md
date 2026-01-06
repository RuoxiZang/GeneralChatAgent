# 通用 Chat Agent

一个具备搜索能力和记忆管理的智能对话助手，基于 DeepInfra API 实现。

## 功能特性

- **🔍 语义搜索工具**：
  - 基于向量嵌入的语义搜索（使用 BGE-M3 模型）
  - 余弦相似度排序，返回最相关的 Top-K 结果
  - 支持从外部 txt 文件加载知识库
- **🧠 记忆系统**：
  - 短期记忆：保留最近对话上下文
  - 长期记忆：自动提取并存储用户偏好信息
- **🤖 智能推理**：ReAct 模式，自主决定是否调用搜索工具

## 快速开始

### 1. 安装依赖

```bash
pip install openai python-dotenv numpy
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
DEEPINFRA_API_KEY=your_actual_api_key_here
DEEPINFRA_BASE_URL=https://api.deepinfra.com/v1/openai
```

### 3. 配置知识库

编辑 `knowledge_base.txt` 文件，每行一条知识：

```
DeepInfra 提供高性价比的 LLM 推理 API 服务。
Python 3.12 引入了更快的解释器和改进的 f-string 解析。
...
```

### 4. 运行

```bash
python agent.py
```

## 使用示例

```
User: 我叫小明,是一个Python程序员
[Memory] 提取到新记忆: 用户姓名：小明，职业：Python程序员
Agent: 你好，小明！很高兴认识你...
    # 主程序
├── knowledge_base.txt    # 知识库文件（每行一条知识）
├── .env                  # 环境配置（需自行创建）
├── .gitignore            # Git 忽略配置
└── README.md             # 项目文档
```

## 核心组件

| 模块 | 说明 |
|------|------|
| `SearchTool` | 基于向量相似度的语义搜索 |
| `MemoryManager` | 对话历史与用户画像管理 |
| `UniversalAgent` | Agent 核心逻辑 |

## 技术栈

- Python 3.8+
- OpenAI SDK
- DeepInfra API (LLM + Embedding)
- NumPy (向量计算)
- python-dotenv (环境变量管理)

## 配置说明

在 `agent.py` 中可调整以下参数：

- `MODEL_NAME`：对话模型（默认 `Meta-Llama-3.1-70B-Instruct`）
- `EMBEDDING_MODEL`：嵌入模型（默认 `BAAI/bge-m3`）
- `max_context_turns`：短期记忆轮数（默认 3）
- `top_k`：搜索返回结果数量（默认 3）

## 注意事项

- 确保 `.env` 文件不被提交到版本控制（已在 `.gitignore` 中配置）
- 首次运行时会为知识库建立向量索引，需要一定时间
- 可通过编辑 `knowledge_base.txt` 动态扩充知识库，重启后生效 Agent 核心逻辑 |

## 技术栈

- Python 3.8+
- OpenAI SDK
- DeepInfra API
- dotenv

## 注意事项

- 确保 `.env` 文件不被提交到版本控制（添加到 `.gitignore`）
- 默认使用 `Meta-Llama-3.1-70B-Instruct` 模型
- 短期记忆窗口默认保留最近 3 轮对话
