# Mini Agent —— 最小但完整的 Agent 系统

一个从零实现的 Agent 系统，用于学习 **LLM → Function Calling → Tool Dispatcher → Tool → LLM** 这条核心链路。

**不使用任何 Agent 框架**（LangChain、LangGraph、AutoGen 等），核心流程全部用 Python 手写。

---

## 项目介绍

Mini Agent 实现了一个最小但完整的智能体：

- **智谱 AI (GLM-4-Flash)** 作为大模型，支持 Function Calling
- **RAG (ChromaDB + 智谱 Embedding)** 作为知识库工具
- **OpenWeather API** 作为天气工具
- **Calculator (AST 安全求值)** 作为计算器工具

Agent 根据用户问题自动判断是否需要调用工具、调用哪个工具：

```
用户：ChromaDB 和 Milvus 有什么区别？
Agent：调用 search_knowledge_base → 获取知识库资料 → 回答

用户：北京现在天气怎么样？
Agent：调用 get_weather → OpenWeather API → 回答

用户：12345 * 6789 等于多少？
Agent：调用 calculate → 返回计算结果

用户：你好
Agent：不调用工具，直接回答
```

---

## Agent 工作流程

```
用户输入
    ↓
智谱 GLM (LLM)
    ↓
LLM 判断是否需要工具
    ↓
┌─ 不需要 → 直接返回回答
└─ 需要   → Function Calling
              ↓
          Tool Dispatcher（根据工具名找到 Python 函数）
              ↓
          执行对应工具
              ↓
          Tool Result
              ↓
          返回给 GLM
              ↓
          GLM 最终回答
```

---

## 上下文管理

对话历史不会无限增长：每轮开始前会检查历史大小，超过预算自动压缩。

### 配置（`app/config.py`）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MAX_CONTEXT_TOKENS` | 6000 | 历史超过该值（估算 token）触发压缩 |
| `KEEP_RECENT_TOKENS` | 3000 | 压缩时为最近对话保留的预算 |

> token 数是估算值（汉字 ≈ 1，其他字符 ≈ 4 个 ≈ 1），不依赖分词库。
> glm-4-flash 实际支持 128K 上下文，这里设小是为了控制成本、方便观察压缩效果。

### 压缩策略（`app/agent/context_manager.py`）

```
历史超过 MAX_CONTEXT_TOKENS
    ↓
按"轮次"分组（一条 user 消息 + 其后的 assistant/tool 消息为一轮）
—— 保证 assistant 的 tool_calls 和对应 tool 结果永不拆散
    ↓
较早轮次 → 用 LLM 生成摘要（以 system 消息保留在历史开头）
最近轮次 → 原样保留（预算 KEEP_RECENT_TOKENS）
    ↓
摘要 API 失败 → 退化为直接丢弃旧轮次（不中断对话）
```

压缩后历史仍然包含早期对话的关键信息（主题、结论、数据），
但"它指什么"这类多轮指代主要依赖最近轮次，因此近期对话完整保留。

### 对话内命令

| 命令 | 作用 |
|------|------|
| `/clear` | 清空上下文，开始全新对话 |
| `/status` | 查看当前上下文用量（消息数 / 估算 token / 预算） |

观察压缩效果：连续进行多轮长对话（或多次 RAG 检索），日志会出现：

```
INFO - Compressing history: 6320 tokens -> old 4 turns summarized, recent 3 turns kept (2890 tokens)
INFO - Generated summary for 4 old turns (256 chars)
INFO - Context compressed: 9 messages, ~3100 tokens now
```

---

## RAG 检索流水线（问题重写 + 重排）

知识库检索不是"拿问题直接查向量库"一步到位，而是一条完整流水线
（`app/rag/retriever.py`）：

```
用户问题（可能带指代、省略）
    ↓
① 问题重写（app/rag/query_rewriter.py）
   结合最近对话历史，把问题改写成独立、明确的查询
   "它和 Milvus 有什么区别？" → "ChromaDB 和 Milvus 有什么区别？"
    ↓
② 向量粗检索（app/rag/vector_store.py）
   用重写后的查询查 ChromaDB，多召回 RAG_CANDIDATE_K 个候选
    ↓
③ 重排（app/rag/reranker.py）
   LLM 给每个候选打 0~10 的相关性分数，按分数精排，取 RAG_TOP_K 个
    ↓
最终检索结果 → 交给 LLM 生成回答
```

### 为什么需要问题重写？

向量检索对"独立、完整"的查询最友好。多轮对话里用户习惯说
"它呢？""这个怎么用？"——直接拿去检索效果很差。
问题重写用一次 LLM 调用，结合最近几条对话消息，把指代
（"它""这个"）替换成具体对象、补全省略的主语。

对话历史由 Agent 循环 → Tool Dispatcher 自动注入给检索工具
（`search_knowledge_base` 声明了 `history` 参数就会收到），
该参数不出现在 LLM 看到的工具 Schema 里。

### 为什么需要重排？

向量相似度衡量的是"语义接近"，不是"能回答问题"。
粗检索先多召回一些候选（`RAG_CANDIDATE_K`），再由 LLM 按
"是否回答了查询"打 0~10 分精排，只把最相关的 `RAG_TOP_K` 个
交给模型。打分要求模型输出 JSON，代码对 ``` 围栏、漏掉的
编号、超范围分数都做了容错。

### 容错设计

两个环节都**绝不阻断检索**：

- 问题重写失败 / 输出为空 → 直接用原始查询检索
- 重排调用失败 / 输出解析失败 → 退回向量检索的原始排序
- 两个开关都可以在 `app/config.py` 里单独关闭

### 配置（`app/config.py`）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `RAG_REWRITE_ENABLED` | `True` | 是否启用问题重写 |
| `RAG_REWRITE_HISTORY_MSGS` | `4` | 重写时参考最近几条对话消息 |
| `RAG_RERANK_ENABLED` | `True` | 是否启用重排 |
| `RAG_CANDIDATE_K` | `8` | 向量粗检索召回的候选数量 |
| `RAG_TOP_K` | `4` | 重排后最终返回的片段数 |

---

## Tool Calling 工作流程

```
LLM 返回 tool_call
      ↓
读取 tool name（如 "get_weather"）
      ↓
在 tools_map 中找到对应 Python 函数
      ↓
解析 arguments（如 {"city": "北京"}）
      ↓
执行函数 → 得到结果
      ↓
构造 tool message
      ↓
再次发送给 LLM
      ↓
LLM 基于工具结果生成最终回答
```

---

## 项目结构

```
mini-agent/
│
├── app/
│   ├── agent/                  # Agent 核心模块
│   │   ├── agent.py            #   Agent 主循环（核心！）
│   │   ├── context_manager.py  #   上下文压缩（token 估算 + 摘要）
│   │   ├── prompts.py          #   系统提示词 + 工具 Schema 定义
│   │   └── tool_dispatcher.py  #   工具分发器（核心！）
│   │
│   ├── tools/                  # 工具实现
│   │   ├── calculator.py       #   安全计算器（AST 求值）
│   │   ├── rag.py              #   RAG 检索工具（调用完整检索流水线）
│   │   └── weather.py          #   天气工具（OpenWeather API）
│   │
│   ├── rag/                    # RAG 内部实现
│   │   ├── loader.py           #   文档加载
│   │   ├── chunker.py          #   文本切分
│   │   ├── embedder.py         #   Embedding 生成
│   │   ├── query_rewriter.py   #   问题重写（结合对话历史改写查询）
│   │   ├── reranker.py         #   重排（LLM 相关性打分精排）
│   │   ├── retriever.py        #   检索流水线（重写 → 粗检索 → 重排）
│   │   └── vector_store.py     #   ChromaDB 向量存储
│   │
│   ├── config.py               # 配置管理（环境变量）
│   ├── llm.py                  # 智谱 AI LLM 调用封装
│   └── logging_config.py       # 日志系统（文件日志 + 自动清理）
│
├── data/
│   ├── documents/
│   │   └── knowledge.txt       # 知识库文档
│   └── chroma_db/              # ChromaDB 持久化（自动生成）
│
├── logs/                       # 日志文件（自动生成，按日期分文件）
│
├── tests/                      # 单元测试
│   ├── test_calculator.py
│   ├── test_context_manager.py
│   ├── test_query_rewriter.py  #   问题重写测试
│   ├── test_reranker.py        #   重排测试
│   ├── test_retriever.py       #   检索流水线测试
│   ├── test_weather.py
│   ├── test_rag.py
│   └── test_tool_dispatcher.py
│
├── .env.example                # 环境变量示例
├── .gitignore
├── requirements.txt
├── main.py                     # 入口
└── README.md
```

---

## 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)（Python 包管理工具）
- 智谱 AI API Key
- OpenWeather API Key

---

## 安装依赖

### 1. 克隆项目并进入目录

```bash
cd mini-agent
```

### 2. 创建虚拟环境并安装依赖

```bash
# 使用 uv 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux / macOS:
source .venv/bin/activate

# 使用 uv 安装依赖
uv pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入真实 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```
ZHIPU_API_KEY=你的智谱API_Key
OPENWEATHER_API_KEY=你的OpenWeather_API_Key
```

---

## API Key 获取

### 智谱 AI API Key

1. 访问 [智谱开放平台](https://open.bigmodel.cn/)
2. 注册/登录账号
3. 在 API Keys 页面创建新的 API Key
4. 复制 Key 到 `.env` 的 `ZHIPU_API_KEY`

### OpenWeather API Key

1. 访问 [OpenWeather](https://openweathermap.org/api)
2. 注册账号
3. 在 API Keys 页面获取 Key（免费版即可）
4. 复制 Key 到 `.env` 的 `OPENWEATHER_API_KEY`

---

## RAG 知识库初始化

首次使用前，需要构建 RAG 知识库索引：

```bash
python main.py --init-rag
```

这会：
1. 读取 `data/documents/knowledge.txt`
2. 将文本切分为 chunk
3. 调用智谱 Embedding API 生成向量
4. 写入 ChromaDB

---

## 启动方式

```bash
python main.py
```

启动后进入交互式对话，输入问题即可。

输入 `quit` 或 `exit` 退出。

注意：日志不再打印到终端，全部写入 `logs/` 目录下的日志文件（文件名带日期，超期自动清理）。

---

## 日志系统

日志**只写入文件，不在终端显示**（保持对话界面干净）。

### 日志文件位置

```
logs/
├── mini-agent_2026-09-27.log    # 一天一个文件，文件名带日期
└── mini-agent_2026-09-28.log
```

- 同一天多次运行共用一个文件（追加写入），每次启动有一条分隔线区分
- 启动横幅会显示本次日志文件路径

### 配置（`app/config.py`）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LOGS_DIR` | `<项目>/logs` | 日志目录 |
| `LOG_LEVEL` | `INFO` | 日志级别：DEBUG / INFO / WARNING / ERROR |
| `LOG_RETENTION_DAYS` | `7` | 日志保留天数，**每次启动自动清理更早的日志文件** |

想调整清理周期，改 `LOG_RETENTION_DAYS` 即可（如 `30` 表示保留一个月）。

### 查看日志

对话过程中发生的一切（工具调用、RAG 检索、上下文压缩、错误堆栈）都在日志文件里：

```bash
# PowerShell: 查看今天的日志
Get-Content logs/mini-agent_2026-09-27.log -Tail 50 -Wait

# Git Bash: 实时跟踪
tail -f logs/mini-agent_2026-09-27.log
```

终端出错的提示（如 `[启动失败] ...`）也会指向日志文件，详细信息以日志为准。

---

## 测试案例

### 测试 1：普通对话（不调用工具）

```
用户: 你好
Agent: 你好！有什么我可以帮助你的吗？
```

### 测试 2：计算器

```
用户: 12345 * 6789 等于多少？
Agent: → 调用 calculate("12345 * 6789")
       → 结果: 83810205
       → Agent 回答: 12345 * 6789 = 83810205
```

### 测试 3：天气

```
用户: 北京现在天气怎么样？
Agent: → 调用 get_weather("北京")
       → OpenWeather API 返回天气数据
       → Agent 回答: 北京目前天气...
```

### 测试 4：RAG 知识库

```
用户: ChromaDB 和 Milvus 有什么区别？
Agent: → 调用 search_knowledge_base("ChromaDB 和 Milvus 有什么区别？")
       → ChromaDB 检索相关文档
       → Agent 基于知识库内容回答
```

### 测试 5：多轮对话

```
用户: 我最近在学习 ChromaDB
Agent: ChromaDB 是一个...

用户: 它和 Milvus 有什么区别？
Agent: → 问题重写: "它和 Milvus 有什么区别？"
         → "ChromaDB 和 Milvus 有什么区别？"（结合 history 消解指代）
       → 向量粗检索召回 8 个候选
       → LLM 重排打分，取最相关的 4 段
       → 回答 ChromaDB 和 Milvus 的区别
```

### 运行单元测试

```bash
python -m pytest tests/ -v
```

---

## 常见错误

### 1. `缺少 ZHIPU_API_KEY`

**原因**：未配置 `.env` 文件或环境变量。

**解决**：在项目根目录创建 `.env` 文件，填入 `ZHIPU_API_KEY=你的key`。

### 2. `缺少 OPENWEATHER_API_KEY`

**原因**：未配置 OpenWeather API Key。

**解决**：在 `.env` 中填入 `OPENWEATHER_API_KEY=你的key`。

### 3. `知识库集合不存在，请先运行 RAG 知识库初始化`

**原因**：未运行 `python main.py --init-rag` 构建索引。

**解决**：先运行 `python main.py --init-rag`。

### 4. `智谱 API 调用失败`

**原因**：API Key 无效、网络问题、或 API 额度用完。

**解决**：
- 检查 API Key 是否正确
- 检查网络连接
- 检查智谱账户余额

### 5. `OpenWeather API Key 无效`

**原因**：API Key 错误或刚创建尚未生效（通常需要等待几分钟）。

**解决**：确认 Key 正确，稍等几分钟后重试。

### 6. `找不到城市 'xxx'`

**原因**：OpenWeather 不认识该城市名。

**解决**：尝试使用英文城市名（如 "Beijing" 而非 "北京"），或确认城市名拼写。

### 7. ChromaDB 版本兼容问题

**原因**：chromadb 版本不兼容。

**解决**：`uv pip install chromadb>=0.5.0`

### 8. 程序出错但终端看不到详细日志

**原因**：日志已改为只写文件，终端只显示对话内容和简要错误提示。

**解决**：打开 `logs/` 目录下今天的日志文件查看完整错误堆栈。

### 9. 日志文件占用空间

**原因**：日志每天一个文件，长期运行会累积。

**解决**：默认保留 7 天，启动时自动清理过期日志。如需调整，修改 `app/config.py` 中的 `LOG_RETENTION_DAYS`。

---

## 核心代码导读

如果你想理解 Agent 是怎么运行的，按这个顺序读代码：

1. **`main.py`** — 入口，看 Agent 如何启动、日志如何初始化
2. **`app/agent/agent.py`** — **核心！** Agent 主循环，看 LLM → Tool → LLM 的完整流程
3. **`app/agent/tool_dispatcher.py`** — **核心！** 工具分发器，看如何根据工具名找到并执行 Python 函数（含对话历史自动注入）
4. **`app/agent/prompts.py`** — 工具定义，看 LLM 看到的 Tool Schema 是什么
5. **`app/agent/context_manager.py`** — 上下文压缩：token 估算、按轮次分组、超限摘要
6. **`app/rag/retriever.py`** — **核心！** 检索流水线：问题重写 → 向量粗检索 → 重排
7. **`app/rag/query_rewriter.py`** — 问题重写：结合对话历史把问题改写成独立查询
8. **`app/rag/reranker.py`** — 重排：LLM 给候选片段打相关性分并精排
9. **`app/llm.py`** — LLM 调用封装
10. **`app/tools/calculator.py`** — 计算器工具实现
11. **`app/tools/weather.py`** — 天气工具实现
12. **`app/tools/rag.py`** — RAG 工具实现（调用检索流水线）
13. **`app/logging_config.py`** — 日志系统：文件日志、按日期分文件、自动清理

---

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | 智谱 AI GLM-4-Flash |
| Embedding | 智谱 AI embedding-3 |
| 向量数据库 | ChromaDB |
| 天气 API | OpenWeather |
| HTTP 客户端 | httpx |
| 配置管理 | python-dotenv |

---

## License

MIT
