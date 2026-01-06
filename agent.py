import os
import json
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv
import numpy as np
# 加载 .env 文件
load_dotenv()

# =================配置区域=================
# 从 .env 文件中读取配置
API_KEY = os.getenv("DEEPINFRA_API_KEY", "YOUR_DEEPINFRA_API_KEY")
BASE_URL = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")
# 推荐使用 Llama-3-70B 或 Qwen-2.5-72B 等指令遵循能力强的模型
MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct" 
EMBEDDING_MODEL = "BAAI/bge-m3"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# =================模块一：搜索工具 (Search Capability)=================
class SearchTool:
    def __init__(self, client: OpenAI, top_k: int = 3, kb_path: str = "knowledge_base.txt"):
        self.client = client
        self.top_k = top_k
        
        # 从本地文件加载知识库
        print(f"[System] 正在从 {kb_path} 加载知识库...")
        self.knowledge_base = self._load_knowledge_base(kb_path)
        print(f"[System] 已加载 {len(self.knowledge_base)} 条知识。")
        
        print("[System] 正在为知识库建立索引（计算向量）...")
        self.doc_vectors = [self._get_embedding(doc) for doc in self.knowledge_base]
        print("[System] 索引建立完成。")
    
    def _load_knowledge_base(self, file_path: str) -> List[str]:
        """从txt文件加载知识库，每行一条知识"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 读取所有行，去除空白行
                lines = [line.strip() for line in f.readlines() if line.strip()]
            return lines
        except FileNotFoundError:
            print(f"[Warning] 未找到知识库文件 {file_path}，使用空知识库。")
            return []
        except Exception as e:
            print(f"[Error] 读取知识库文件失败: {e}")
            return []
        
    def _get_embedding(self, text: str) -> List[float]:
        """调用 API 获取文本向量"""
        try:
            # 注意：不同模型对换行符敏感，通常建议替换为空格
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(
                input=[text],
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[Embedding Error] {e}")
            return []

    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str) -> str:
        # """简单的关键词匹配搜索"""
        # print(f"\n[System] 正在搜索: {query} ...")
        # results = []
        # # print("[Debug] Knowledge Base:", self.knowledge_base)
        # for doc in self.knowledge_base:
        #     if any(word in doc for word in query.split()):
        #         results.append(doc)
        # # print(f"[System] 搜索完成，找到 {len(results)} 条结果。\n")
        # if results:
        #     return "搜索结果:\n" + "\n".join([f"- {r}" for r in results])
        # else:
        #     return "搜索结果: 未找到相关本地信息。"
        """基于向量相似度的语义搜索"""
        print(f"\n[System] 正在进行语义搜索: {query} ...")
        
        # 1. 获取查询词的向量
        query_vector = self._get_embedding(query)
        if not query_vector:
            return "搜索服务暂时不可用。"

        # 2. 计算与知识库中每一条的相似度
        scores = []
        for doc_vec in self.doc_vectors:
            score = self._cosine_similarity(query_vector, doc_vec)
            scores.append(score)

        # 3. 排序并获取索引 (从高到低)
        # argsort 返回的是从小到大的索引，所以我们取最后 top_k 个并反转
        top_indices = np.argsort(scores)[-self.top_k:][::-1]

        results = []
        # 设定一个相似度阈值，太低的相关性不如不返回 (例如 0.4)
        threshold = 0.4 
        
        for idx in top_indices:
            score = scores[idx]
            if score > threshold:
                results.append(f"{self.knowledge_base[idx]} (相似度: {score:.2f})")
        
        if results:
            return "搜索结果(按相关性排序):\n" + "\n".join([f"- {r}" for r in results])
        else:
            return "搜索结果: 未找到足够相关的本地信息。"

# =================模块二：记忆系统 (Memory System)=================
class MemoryManager:
    def __init__(self, max_context_turns=3):
        self.short_term_memory: List[Dict] = []  # 对话历史
        self.long_term_memory: List[str] = []    # 用户画像/偏好
        self.max_context_turns = max_context_turns # 限制最近N轮

    def add_message(self, role: str, content: str):
        """添加对话到短期记忆"""
        self.short_term_memory.append({"role": role, "content": content})

    def get_context_window(self) -> List[Dict]:
        """获取最近 N 轮对话作为 Prompt 上下文"""
        # 保留 System prompt (如果有) 和最近的交互
        return self.short_term_memory[-(self.max_context_turns * 2):]

    def extract_and_save_user_info(self, last_user_input: str):
        """
        记忆管理核心：
        使用 LLM 分析用户输入，提取用户偏好/背景存入长期记忆。
        """
        prompt = f"""
        请分析以下用户输入。如果包含用户的个人信息（姓名、职业、喜好、计划等），请简练地提取出来。
        如果没有包含个人信息，请输出 "None"。
        
        用户输入: "{last_user_input}"
        """
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            # print(f"[Memory Extraction] LLM 输出: {content}")
            if content != "None" and "None" not in content:
                print(f"[Memory] 提取到新记忆: {content}")
                self.long_term_memory.append(content)
        except Exception as e:
            print(f"[Memory Error] {e}")

    def get_long_term_memory_str(self) -> str:
        """记忆读取：格式化长期记忆供 System Prompt 使用"""
        if not self.long_term_memory:
            return "暂无用户偏好记录。"
        return "\n".join([f"- {m}" for m in self.long_term_memory])

# =================模块三：Agent 核心 (The Agent)=================
class UniversalAgent:
    def __init__(self):
        self.memory = MemoryManager(max_context_turns=3)
        self.search_tool = SearchTool(client)

    def _construct_system_prompt(self) -> str:
        """构建动态 System Prompt，注入长期记忆和工具说明"""
        user_profile = self.memory.get_long_term_memory_str()
        
        return f"""
        你是一个智能助手。
        
        【能力说明】
        1. 你拥有搜索工具。如果你不知道问题的答案，或者需要查询实时/事实性信息，请输出特定指令：[SEARCH: 关键词]
        2. 如果不需要搜索，请直接回答用户。
        
        【记忆库】
        这是你关于该用户的记忆，请在回答时参考：
        {user_profile}
        """

    def chat(self, user_input: str):
        # 1. 记忆管理：先尝试提取本轮对话中的用户信息
        self.memory.extract_and_save_user_info(user_input)
        
        # 2. 准备上下文
        self.memory.add_message("user", user_input)
        # print("[Debug] Short Term Memory: [",self.memory.short_term_memory)
        context = self.memory.get_context_window()
        # print("[Debug] Context:", context)
        system_prompt = self._construct_system_prompt()
        
        messages = [{"role": "system", "content": system_prompt}] + context

        # 3. 第一轮推理：决定是否使用工具或直接回答
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7
        )
        ai_content = response.choices[0].message.content
        print(f"[Debug] AI First Response: {ai_content}")
        # 4. 判断是否触发搜索 (简单的 ReAct 模拟)
        if "[SEARCH:" in ai_content:
            # 解析查询词
            try:
                start = ai_content.index("[SEARCH:") + 8
                end = ai_content.index("]", start)
                query = ai_content[start:end].strip()
                print(f"[System] 触发搜索工具，查询关键词: {query}")
                # 执行搜索
                search_result = self.search_tool.search(query)
                print(f"[System] search返回结果: {search_result}")
                # 5. 将搜索结果追加到临时对话中，进行二次生成
                messages.append({"role": "assistant", "content": ai_content})
                messages.append({"role": "user", "content": f"工具调用结果: {search_result}\n请根据这个结果回答用户的问题。"})
                # print(f"[Debug] Messages for second round: {messages}")
                final_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.3
                )
                final_answer = final_response.choices[0].message.content
                
                
                print(f"Agent: {final_answer}")
                self.memory.add_message("assistant", final_answer)
                
            except ValueError:
                # 格式解析失败，直接输出原始内容
                print(f"Agent: {ai_content}")
                self.memory.add_message("assistant", ai_content)
        else:
            # 直接回答
            print(f"Agent: {ai_content}")
            self.memory.add_message("assistant", ai_content)

# =================主程序入口=================
if __name__ == "__main__":
    if API_KEY == "YOUR_DEEPINFRA_API_KEY":
        print("请先在代码中配置 DeepInfra API KEY")
    else:
        agent = UniversalAgent()
        print("======= 通用 Agent 已启动 (输入 'exit' 退出) =======")
        print("尝试输入：\n1. 'DeepInfra是做什么的？' (测试搜索)\n2.'我叫小明，是一个Python程序员' (测试记忆存储) \n3. '结合我的职业，DeepInfra对我有什么用？' (测试记忆读取+上下文)")
        
        while True:
            user_in = input("\nUser: ")
            if user_in.lower() in ["exit", "quit"]:
                break
            agent.chat(user_in)