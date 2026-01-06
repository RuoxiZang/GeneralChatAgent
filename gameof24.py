import itertools
import math
from typing import List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import deque

# ==========================================
# 核心 ToT 框架
# ==========================================

@dataclass
class ThoughtNode:
    """思维树节点"""
    state: List[Tuple[float, str]]  # 存储 [(数值, 表达式), ...]
    parent: Optional['ThoughtNode'] = None
    operation: str = ""             # 产生该节点的描述（调试用）
    score: float = 0.0
    depth: int = 0
    
    def __repr__(self):
        return f"Node(depth={self.depth}, val={[x[0] for x in self.state]})"

class TreeOfThoughts:
    """
    通用 Tree of Thoughts 搜索框架
    """
    def __init__(
        self,
        thought_generator: Callable[[ThoughtNode], List[ThoughtNode]],
        state_evaluator: Callable[[ThoughtNode], float],
        goal_checker: Callable[[ThoughtNode], bool]
    ):
        self.thought_generator = thought_generator
        self.state_evaluator = state_evaluator
        self.goal_checker = goal_checker
        self.visited_states: Set[str] = set()

    def _get_state_signature(self, state: List[Tuple[float, str]]) -> str:
        """生成状态指纹用于去重：对数值进行排序并格式化"""
        # 只取数值部分进行排序，保留2位小数精度避免浮点误差导致的重复不识别
        values = sorted([round(x[0], 5) for x in state])
        return str(values)

    def search(self, initial_state: Any, strategy: str = 'bfs', beam_width: int = 100) -> Optional[ThoughtNode]:
        """
        执行搜索
        :param strategy: 'bfs' or 'dfs'
        :param beam_width: 仅用于 beam search (此处简化未实现完整beam，用bfs替代)
        """
        root = ThoughtNode(state=initial_state, score=0.5, depth=0)
        self.visited_states.clear()
        
        # 根据策略选择容器
        if strategy == 'bfs':
            queue = deque([root])
        elif strategy == 'dfs':
            queue = [root]  # 使用列表作为栈
        else:
            raise ValueError("Unsupported strategy")

        while queue:
            # BFS: popleft(), DFS: pop()
            current_node = queue.popleft() if strategy == 'bfs' else queue.pop()
            
            # 1. 检查是否达成目标
            if self.goal_checker(current_node):
                return current_node
            
            # 2. 生成新的想法 (Thoughts)
            # 剪枝逻辑：如果已经递归太深或者数字用完了但没结果，就不再生成
            if len(current_node.state) <= 1:
                continue

            children = self.thought_generator(current_node)
            
            # 3. 评估与剪枝
            valid_children = []
            for child in children:
                # 去重检查
                sig = self._get_state_signature(child.state)
                if sig in self.visited_states:
                    continue
                
                # 评估
                score = self.state_evaluator(child)
                child.score = score
                
                # 简单剪枝：如果评分0（代表死胡同或错误），抛弃
                if score > 0:
                    self.visited_states.add(sig)
                    valid_children.append(child)
            
            # 将子节点加入搜索队列
            # 对于 DFS，逆序加入可以保持生成顺序的直观性（可选）
            if strategy == 'dfs':
                queue.extend(reversed(valid_children))
            else:
                queue.extend(valid_children)
                
        return None

# ==========================================
# 24点 业务逻辑实现
# ==========================================

class Point24Solver:
    """24点求解器"""
    
    def __init__(self):
        self.tot = TreeOfThoughts(
            thought_generator=self.generate_thoughts,
            state_evaluator=self.evaluate_state,
            goal_checker=self.check_goal
        )
        self.epsilon = 1e-5 # 浮点数比较容差

    def _ops(self, a_val, a_expr, b_val, b_expr):
        """生成两个数字的所有运算结果"""
        res = []
        # 加法 (满足交换律，在 generator 中控制顺序，这里只做一次)
        res.append((a_val + b_val, f"({a_expr} + {b_expr})"))
        
        # 乘法
        res.append((a_val * b_val, f"({a_expr} * {b_expr})"))
        
        # 减法 (不满足交换律，需要两种情况)
        res.append((a_val - b_val, f"({a_expr} - {b_expr})"))
        res.append((b_val - a_val, f"({b_expr} - {a_expr})"))
        
        # 除法 (需检查分母)
        if abs(b_val) > self.epsilon:
            res.append((a_val / b_val, f"({a_expr} / {b_expr})"))
        if abs(a_val) > self.epsilon:
            res.append((b_val / a_val, f"({b_expr} / {a_expr})"))
            
        return res

    def generate_thoughts(self, node: ThoughtNode) -> List[ThoughtNode]:
        """
        思维生成器：从当前数字列表中任选2个进行运算，生成下一层节点
        """
        current_state = node.state
        children = []
        n = len(current_state)
        
        # 遍历所有两两组合 C(n, 2)
        for i in range(n):
            for j in range(i + 1, n):
                val1, expr1 = current_state[i]
                val2, expr2 = current_state[j]
                
                # 获取这两数运算后的所有可能结果
                new_values = self._ops(val1, expr1, val2, expr2)
                
                # 剩余未参与运算的数字
                remaining = [current_state[k] for k in range(n) if k != i and k != j]
                
                for res_val, res_expr in new_values:
                    # 新状态 = 剩余数字 + 运算结果
                    new_state = remaining + [(res_val, res_expr)]
                    
                    # 创建子节点
                    child_node = ThoughtNode(
                        state=new_state,
                        parent=node,
                        depth=node.depth + 1
                    )
                    children.append(child_node)
        return children

    def evaluate_state(self, node: ThoughtNode) -> float:
        """评估函数：判断状态是否有潜力或已成功"""
        # 如果只剩一个数字，检查是否是24
        if len(node.state) == 1:
            val = node.state[0][0]
            if abs(val - 24.0) < self.epsilon:
                return 1.0  # 完美解
            else:
                return 0.0  # 失败的死胡同
        
        # 中间状态，简单给予 0.5 分（在此简单逻辑中，不进行复杂的数学启发式剪枝）
        return 0.5

    def check_goal(self, node: ThoughtNode) -> bool:
        """目标检查"""
        return self.evaluate_state(node) == 1.0

    def solve(self, numbers: List[int], strategy: str = 'bfs') -> Optional[str]:
        """
        求解主入口
        """
        # 初始化状态：[(数值, "数值字符串"), ...]
        initial_state = [(float(x), str(x)) for x in numbers]
        
        # 执行搜索
        result_node = self.tot.search(initial_state, strategy=strategy)
        
        if result_node:
            # 提取最终表达式 (去一层外括号)
            final_expr = result_node.state[0][1]
            if final_expr.startswith("(") and final_expr.endswith(")"):
                final_expr = final_expr[1:-1]
            return f"{final_expr} = 24"
        return None

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    solver = Point24Solver()
    
    test_cases = [
        [3, 3, 8, 8],  # 经典难点 (8/(3-8/3))
        [1, 1, 1, 1],  # 无解
        [1, 2, 3, 4],  # 简单
        [5, 5, 5, 1],  # (5-1/5)*5
    ]
    
    print(f"{'Input':<15} | {'Strategy':<5} | {'Result'}")
    print("-" * 50)
    
    for nums in test_cases:
        # 测试 BFS
        res_bfs = solver.solve(nums, strategy='bfs')
        print(f"{str(nums):<15} | BFS   | {res_bfs}")
        
        # 测试 DFS
        res_dfs = solver.solve(nums, strategy='dfs')
        print(f"{str(nums):<15} | DFS   | {res_dfs}") 