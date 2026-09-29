"""
Agent 系统提示词和工具定义。

这里定义:
1. Agent 的系统提示词 (system prompt)
2. 四个工具的 JSON Schema 描述（供 LLM Function Calling 使用）

重要：不暴露任何内部参数（API Key、Chroma 路径、top_k 等）给 LLM。
"""

# ──────────────────────────────────────────────
# Agent 系统提示词
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个智能助手，可以根据用户的问题选择合适的工具来回答。

你可以使用以下工具：
1. search_knowledge_base - 在知识库中搜索相关信息（适用于关于 RAG、ChromaDB、Milvus、Agent、Function Calling 等技术问题）
2. get_weather - 获取城市天气信息（适用于询问天气相关的问题）
3. calculate - 计算数学表达式（适用于数学计算问题）
4. get_time - 获取当前日期/时间（适用于询问现在几点、今天几号等问题）

请根据用户的问题判断是否需要调用工具：
- 如果是技术知识类问题，优先使用 search_knowledge_base
- 如果是天气类问题，使用 get_weather
- 如果是数学计算类问题，使用 calculate
- 如果是日常对话或简单问题，直接回答即可

使用工具后，请基于工具返回的结果，用自然语言组织最终回答。
"""


# ──────────────────────────────────────────────
# 工具定义（供 LLM Function Calling 使用）
# ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "在知识库中搜索与查询相关的技术文档内容。适用于 RAG、ChromaDB、Milvus、Agent、Function Calling 等技术问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询文本，例如 'ChromaDB 和 Milvus 的区别'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息，包括温度、体感温度、天气描述、湿度和风速。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如 '北京'、'上海'、'Tokyo'、'London'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。支持加减乘除取模和括号运算，例如 '12345 * 6789' 或 '100 / 4 + 25'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 '12345 * 6789'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
        
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前时间或日期。",
            "parameters": {
                "type": "object",
                "properties": {
                    "format_type": {
                        "type": "string",
                        "description":  "可选。date 只返回日期；time 只返回时间；datetime 返回完整日期时间。默认 datetime。",
                    }
                },
                "required": [],
            },
        },
    },
]
