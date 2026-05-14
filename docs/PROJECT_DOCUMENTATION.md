# Deep Research MCP — 完整项目文档

> 文档版本：v4.0
> 适用：当前 main 分支（含 4 项功能成熟度升级 + 工程化加固后的版本）
> 目标读者：只看这份文档，就能从零理解项目在做什么、为什么这样做、每一块怎么实现

---

## 怎么阅读这份文档

文档分为 13 章，建议第一次读按顺序看：

1. 它是什么 / 解决什么问题（先建立直觉）
2. 关键术语词典（**专业概念在这里全部解释清楚**）
3. 一张图看完整体架构
4. 用户视角的完整生命周期（不看代码，看流程）
5-12. 逐个模块深入（数据流、亮点设计）
13. 配置、运维、部署

如果你已经知道 MCP / LangGraph 是什么，可以跳过第 2 章。如果你只想知道"这个项目跟其他 deep research 有什么不一样"，看第 3 章和第 11 章（亮点设计）就够。

---

## 第 1 章 它是什么 —— 用菜市场比喻先建立直觉

### 1.1 一句话定义

Deep Research MCP 是一个**自动化研究流水线**：你给它一个开放式问题（比如「2025 年 RISC-V 在数据中心的进展」），它会自己拆任务、自己搜索、自己读资料、自己判断证据够不够、自己写一份带引用的研究报告给你。

### 1.2 类比：人类研究员是怎么干活的？

想象一个研究生接到导师给的题目"调研 X 领域 2025 进展"，他不会立刻打开 Word 开始写。他会：

1. **想一下**：这题大概要查哪些方向？要看综述还是看论文？要找新闻还是找数据？（**规划**）
2. **试探搜一下**：先用几个关键词搜搜看，目前网上都有什么资料？（**侦察**）
3. **跟导师对一下**：「我打算这样查，您看 OK 不」，得到点头后再正式开干。（**确认**）
4. **分头查**：把题目拆成几个子方向，分别找资料。（**并行检索**）
5. **整理证据**：把找到的东西做成卡片，每张卡片一个论点 + 来源 + 引文。（**结构化**）
6. **盘一下还缺啥**：「我对 X 子方向掌握得不够，得再查」。（**反思**）
7. **写报告**：按"先讲背景、再讲技术、最后讲展望"的结构组织成文，每个论断后面挂引用。（**写作**）

**这个项目把这套流程做成了一个 AI 系统**。不同的是：人类做完这 7 步要好几天，系统做完要几分钟。

### 1.3 这个项目跟"让 LLM 直接写报告"有什么不一样？

直接让 LLM 写报告：
- **幻觉问题**：模型记错数据、编造来源
- **信息过期**：模型只知道训练时间之前的事
- **无法验证**：你看到结论但不知道是哪儿来的

这个项目：
- 所有论断都强制基于**实际搜索回来的真实网页/论文**
- 每个论断都有可追溯的来源**链接**
- 通过"封闭证据 ID 集 + verbatim quote 校验"**结构性消除**模型编造引用号的可能性（详见第 11 章亮点④）

### 1.4 它和 Perplexity / Gemini Deep Research 是同一类东西

是同一类。不同的是：
- 那些是 SaaS 产品，你用它们的 UI
- 这个项目是开源 + 自部署的 **MCP 服务器**，可以挂到你自己的 AI 助手里（Claude Desktop、Claude Code、Cursor、Codex 等），用你自己的 API key、查你想查的数据源、跑在你自己电脑上

---

## 第 2 章 关键术语词典（外行专用）

第一次遇到的专业名词都在这一章解释。读完后面章节出现这些词时，可以随时回来查。

### 2.1 MCP（Model Context Protocol）

**大白话**：让 AI 助手（比如 Claude Desktop）能调用外部工具的一套标准协议。

**类比**：好比 USB 接口标准。任何符合 MCP 协议的工具，都能"插"到任何符合 MCP 协议的 AI 助手上，不需要双方互相适配。

**这个项目里**：本项目暴露 7 个 MCP 工具（`draft_research_plan`、`start_research_task` 等），AI 助手可以像调用函数一样调用它们。

### 2.2 LLM 与 LiteLLM

**LLM**：Large Language Model，大语言模型，比如 GPT-4、Claude、DeepSeek 都属于。

**LiteLLM**：一个 Python 库。它的价值是：**统一不同厂商 LLM 的调用接口**。你写代码时只需要写 `litellm.acompletion(model="xxx/yyy", messages=[...])`，LiteLLM 自动判断 `xxx` 是 deepseek / openai / anthropic 还是其他，去调对应厂商的 API。

**这个项目里**：所有 LLM 调用都走 LiteLLM。你换 LLM 厂商只要改一个 env 变量，**代码一行都不用动**。

### 2.3 LangGraph

**大白话**：一个让你"画流程图"来定义 AI 工作流的 Python 库。

**类比**：好比施工蓝图。你不直接告诉机器"先干啥再干啥"，而是画一个图：节点 A → 节点 B → 节点 C，然后把这张图交给 LangGraph 帮你执行。

**优点**：图里的每个节点可以独立测试、可以并行执行、可以中途回到上游节点（循环），整个图的状态自动持久化（断电可以恢复）。

**这个项目里**：整个研究流程是一张 LangGraph 图，节点包括 supervisor、section_researcher、reflector、writer 等。

### 2.4 State / Reducer

**State**：图执行时的"全局状态"，所有节点共享。本项目的 state 叫 `ResearchState`，里面有 task_id、topic、所有收集到的卡片、当前进展等。

**Reducer**：当多个并行节点同时修改 state 的同一个字段时，告诉 LangGraph 怎么合并。比如 10 个并行的 researcher 都在往 `knowledge_cards` 里加卡片，reducer 说"把所有列表 append 起来"。

### 2.5 Send API（Map-Reduce）

**大白话**：LangGraph 的并行机制。

**类比**：组长把任务分给 N 个组员同时做（Map），然后等他们都做完再汇总（Reduce）。

**这个项目里**：5 个 section 同时让 5 个 `section_researcher` 节点并行干活，等它们都完成再汇总到 `collect_results` 节点。

### 2.6 Embedder（嵌入器）与 Reranker（重排序器）

**Embedder**：把一段文字变成一个数字向量（比如 384 维）。两段意思相近的文字，向量也接近。**主要用途**：把搜索结果切成句子粒度，跟用户查询比相似度，找出最相关的那些句子。

**Reranker**：第二轮筛选。Embedder 第一轮筛出 top 100，Reranker 把这 100 条逐个跟查询做更精细的相关性打分，重排出 top 10。

**类比**：Embedder 像广播找人（"会唱歌的请到 3 楼"，100 人响应），Reranker 像 KTV 一对一面试（挑出最好的 10 个）。

**这个项目里**：用本地下载的 `bge-small-zh-v1.5`（embedder）和 `bge-reranker-base`（reranker）。两个模型加载较慢，所以放在**独立子进程**里跑（详见第 9 章 Worker 子进程）。

### 2.7 RAG（Retrieval Augmented Generation）

**大白话**：让 LLM 在回答前先去查资料，然后基于资料生成答案。

**这个项目里**：完整就是一个 RAG 系统——检索阶段（搜索引擎 + 嵌入器 + 重排序器）+ 生成阶段（writer）。

### 2.8 SQLite / aiosqlite / WAL

**SQLite**：一个轻量级数据库，整个数据库就是一个文件（不需要装数据库服务）。

**aiosqlite**：SQLite 的 async 版本，让 Python 的 async 程序能用 SQLite 而不阻塞事件循环。

**WAL（Write-Ahead Logging）**：SQLite 的一种工作模式。开启后允许"多个读 + 一个写同时进行"，吞吐量大幅提升。

**这个项目里**：所有任务的状态（哪些任务跑过、卡片、报告版本）都存在 SQLite 里。开启了 WAL 模式以支持并发。

### 2.9 Checkpoint（图执行检查点）

**大白话**：LangGraph 在每个节点执行完之后，把当前 state 存一份到磁盘。

**作用**：
- 任务跑到一半进程崩了 → 重启后从最后一个 checkpoint 继续
- 用户想做"追问"（follow-up）→ 直接从原任务最后一个 state 拉起来，复用之前的卡片

**这个项目里**：checkpoint 用单独的 SQLite 文件存储。

### 2.10 Human-in-the-loop

**大白话**：流程中间有一步要人工确认。

**这个项目里**：draft（草拟策略）和 start（正式执行）是分开两个 MCP 工具——用户先看 draft 输出的研究策略，觉得 OK 才调用 start。

### 2.11 知识卡片（KnowledgeCard）

**大白话**：把一段网页内容压缩成的结构化记录，包含：
- claim（论点，一句话）
- evidence_summary（佐证摘要）
- exact_excerpt（原文中支持该论点的具体那一句）
- source（来源 URL）
- confidence（置信度：high / medium / low）
- stance（立场：supporting / counter / neutral / limitation）

**类比**：考研做笔记时的索引卡片。一张卡片 = 一个论点 + 一处证据 + 一个出处。

---

## 第 3 章 整体架构一张图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   你（在 Claude Desktop / Codex / Cursor 等 MCP 客户端里输入：              │
│       "调研 2025 年 RISC-V 在数据中心的进展"）                              │
│                                                                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ MCP 协议 (stdio)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MCP Server 层 (deep_research_mcp.py + tools.py)                            │
│  暴露 7 个工具：draft / start / status / result / follow_up / compare / check│
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DeepResearchService 编排层 (service.py)                                    │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │   Settings   │  │ TaskRegistry│  │SearchService│  │  ModelWorker    │    │
│  │ (全部配置)   │  │   Store     │  │ (双层检索)  │  │ (子进程: bge)   │    │
│  └──────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘    │
│         注入到下面所有 Agent 中（AgentContext）                             │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ build_graph()
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LangGraph 工作流 (graph.py)                                                │
│                                                                             │
│        ┌───────────────┐                                                    │
│        │  Supervisor   │ ◄─── 把策略拆成可执行子任务                        │
│        └───────┬───────┘                                                    │
│                ▼                                                            │
│        ┌───────────────┐                                                    │
│        │   Dispatch    │ ──── fan_out_sections (Send API map-reduce)        │
│        └───────┬───────┘                                                    │
│                ▼                                                            │
│   ┌─────────────────────────────────────────┐                              │
│   │  Section Researcher × N (并行执行)      │ ◄── 每个 section 独立干活    │
│   │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │                              │
│   │  │ Sec1 │ │ Sec2 │ │ Sec3 │ │ ...  │    │                              │
│   │  └──────┘ └──────┘ └──────┘ └──────┘    │                              │
│   └─────────────────┬───────────────────────┘                              │
│                     ▼                                                       │
│           ┌───────────────────┐                                             │
│           │ Collect Results   │ ◄── 全局去重，把所有 section 的卡片合并    │
│           └────────┬──────────┘                                             │
│                    ▼                                                        │
│           ┌───────────────────┐                                             │
│           │    Reflector      │ ◄── 评估覆盖度 + 计算饱和度                 │
│           └────────┬──────────┘                                             │
│             ┌──────┴───────┐                                                │
│             │              │                                                │
│       不够 ▼              ▼ 够了                                            │
│      回 Dispatch    Outline Builder ◄── 大纲由证据驱动                      │
│      (循环补研究)         │                                                 │
│                           ▼                                                 │
│                  ┌────────────────────┐                                     │
│                  │ Detect Conflicts   │ ◄── 跨源分歧主动检测                │
│                  └─────────┬──────────┘                                     │
│                            ▼                                                │
│                  ┌────────────────────┐                                     │
│                  │      Writer        │ ◄── 结构化 grounded 生成报告        │
│                  │ (closed-set IDs)   │                                     │
│                  └─────────┬──────────┘                                     │
│                            ▼                                                │
│                          END                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
            产出：Markdown 报告 + 知识卡片 JSON + 来源 JSON + 元数据 + 引用审计
```

### 这张图的关键设计

- **核心 4 节点**（Supervisor / Researcher / Reflector / Writer）担起主流程的语义
- **辅助节点**（Dispatch / Collect / Outline / Detect Conflicts）只做"管道工"工作
- **循环**只发生在 Reflector → Dispatch 这一条边（其他都是单向）
- **并行**只发生在 Researcher 层（map-reduce）
- **持久化**全程都在做（每个节点执行完写一次 checkpoint）

---

## 第 4 章 用户视角的完整生命周期

这一章不讲代码细节，只讲"用户做了 X，系统按顺序做了什么"。

### 4.1 用户在 Claude Desktop 里输入一个研究问题

**用户操作**：
> 帮我用 deep-research 调研 2025 年 RISC-V 在数据中心的进展。

**系统连锁反应**：

1. **Claude Desktop 识别意图**，决定调用 MCP 工具 `draft_research_plan`，把"2025 年 RISC-V 在数据中心的进展"作为 `topic` 参数传过来。

2. **MCP Server 收到请求**，路由到 `tools.tool_draft_plan` 这个方法。

3. **输入校验**：检查 topic 长度（不能超 2000 字符）、不能为空。

4. **生成 task_id**：随机 8 位短 ID，比如 `a3b8c0d1`。

5. **侦察阶段**（planner.generate_seed_queries）：
   - 不是直接让 LLM 写大纲（在还没查任何东西的情况下，LLM 写出来的大纲是"想象出来"的，跟真实资料对不上）
   - 而是让 LLM 生成 3-5 个**侦察用的搜索词**（seed queries），比如：
     - "RISC-V data center 2025"
     - "RISC-V server adoption recent"
     - "RISC-V cloud computing trend"

6. **轻量侦察搜索**（search_service.reconnaissance）：
   - 用上面那 3-5 个 seed queries，去 Tavily（默认搜索引擎）各搜一次
   - 每个 query 只取前 3 条结果
   - 目的：**快速摸清"网上目前都有什么资料"**

7. **草拟执行策略**（planner.draft_execution_plan）：
   - 把侦察拿到的资料 + 用户问题一起喂给 LLM
   - 让 LLM 输出一个**ResearchExecutionPlan**：
     - task_type（任务类型，比如 trend_analysis）
     - source_strategy（用通用搜索还是垂直学术搜索）
     - query_strategy（具体要查哪 4-6 个研究方向，每个方向几个查询）
     - quality_rules（怎么筛选证据）
     - expected_deliverable（最终交付什么）

8. **保存草稿**：所有内容存到 SQLite，task_id 关联。

9. **返回给用户**：把策略渲染成 Markdown 返回给 Claude Desktop。用户看到的是类似：

> 任务 ID：a3b8c0d1
> 已生成调研策略：
> - 研究方向 1：芯片厂商进展（搜 SiFive / Tenstorrent / Ventana）
> - 研究方向 2：数据中心实际部署案例
> - 研究方向 3：性能 benchmark 对比 ARM/x86
> - …
> 请回复"approve"启动正式调研，或修改策略后再确认。

**为什么要这么做**：用户在花掉真正的搜索和写作成本（几十次 LLM 调用 + 几百次搜索）之前，先确认方向对不对。**这是项目最重要的"省钱省时间"机制**——避免系统理解错方向，跑完才发现要返工。

---

### 4.2 用户确认后启动正式调研

**用户操作**：
> approve

**系统连锁反应**：

10. **MCP Server 收到请求**，路由到 `tools.tool_execute_plan`。

11. **加载草稿**：从 SQLite 拉出 `task_id=a3b8c0d1` 的草稿。

12. **启动 model worker**：这是个独立子进程，加载 bge embedder 和 reranker 模型（大约需要 30-60 秒首次加载）。如果已经在跑就跳过。

13. **后台启动 LangGraph 流程**：开一个 asyncio Task，调用 `service._run_graph_stream(initial_state, config)`。**MCP 工具立刻返回**"任务已启动"给用户，不等图跑完。

14. **Supervisor 节点**：
    - 输入：approved_plan + execution_plan
    - 处理：把 `query_strategy` 的每个研究方向展开成 SubTask 列表。每个 SubTask 包含 query、intent、section_id 等
    - 比如：研究方向 1 展开成 3 个 SubTask（每个 SubTask 是一个具体的搜索查询）
    - 输出：sub_tasks（待执行的搜索任务列表）

15. **Dispatch + Map（fan_out_sections）**：
    - 输入：sub_tasks
    - 按 section_id 把任务分组
    - 用 Send API 给每个 section 派一个 `section_researcher` 节点
    - 输出：N 个 Send 对象（N = section 数）

16. **Section Researcher × N 并行执行**：每个 section 独立做以下事情：
    - 对该 section 内的每个 SubTask：
      - **Tier 1 搜索**：用所有 rewritten_queries 同时搜索，指数退避重试
      - **Tier 2 查询重写**：如果 Tier 1 拿到 0 结果，1 次 LLM 调用产生 3 个不同策略的备选 query（SIMPLIFY / SYNONYMS / DECOMPOSE），顺序试到第一个有结果
      - **Tier 3 降级**：还是不行，标记 status=degraded，不阻塞整体
    - 拿到的网页全部送给 model worker 做 embedding + 切片 + 重排序，留下最相关的几十个句子作为 evidence
    - 让 LLM 把 evidence 抽取成 KnowledgeCard（claim + evidence_summary + exact_excerpt + ...）
    - 用 KnowledgeCache 做去重（URL+claim 哈希）
    - 输出：knowledge_cards（这个 section 的卡片列表）+ task_updates（SubTask 状态）

17. **Collect Results**：
    - 所有 section 的卡片自动汇总（LangGraph 的 reducer 把 N 个列表 append 起来）
    - 全局做一遍跨 section 的去重
    - 输出：合并后的 knowledge_cards、loop_count+1

18. **Reflector 评估**：
    - 对每个 section 算质量分（覆盖度、证据数、来源多样性、主要来源占比）
    - 让 LLM 做语义评审（"这些证据真的能回答 section 提出的问题吗？还缺什么？"）
    - 算**饱和度**：综合"本轮新增证据带来多少覆盖度提升 + 新增卡片占比"，得出一个 0-1 的分数
    - 决定：
      - 饱和度 ≥ 0.85 → 走 outline_builder
      - 还有大量缺口 + 没到最大循环数 → 生成补研究任务，回 Dispatch
      - 达到最大循环数（默认 3 轮）→ 强制走 outline_builder
    - 输出：route_to + 可能的新 sub_tasks

19. **Outline Builder**：
    - 输入：section_digests（每个 section 的压缩证据包）+ 执行策略
    - 让 LLM 基于**真实证据**生成最终报告大纲（"先讲背景、再讲技术现状、再讲应用案例、最后讲展望"这种）
    - 输出：plan_data，里面包含 sections 列表，每个 section 引用一组 evidence_digest_ids

20. **Detect Conflicts**（新增节点）：
    - 对每个 section 的卡片做"两两候选对生成"
    - 过滤：同源对去掉、不同立场或同实体不同数值的留下、时间段不重叠的去掉
    - 让 LLM 一次性判断这些对（COMPATIBLE / PARTIAL / CONTRADICTORY）
    - 用代码（不是 LLM）算严重度
    - 输出：section_conflicts（稀疏字典，没分歧的 section 不出现）

21. **Writer 写报告**：
    - 对每个 section：
      - 给 LLM 一个**封闭的 evidence ID 集合**（E1, E2, ..., EN）
      - 让 LLM 输出**结构化 JSON**：每个段落明确声明引用了哪些 evidence_ids
      - 系统校验：超集 ID 丢弃 / 数值必须在引用 excerpt 里出现 / 有 quote 字段必须是引用 excerpt 的 token 子集
      - 系统**自己渲染**引用号 `<sup><a href="url">[N]</a></sup>`——LLM 物理上无法编造引用号
      - 如果 JSON 解析失败，降级到老的自由 markdown 路径
      - 把每个 section 的 audit 记下来
    - 拼接成完整报告：标题 + 导言 + 各 section + 直接回答 + 未来展望 + References 列表
    - 输出：final_report.md 文件 + citation_audit 摘要

22. **任务完成**：lifecycle 改成 "completed"，写所有 artifacts（cards.json、sources.json、metadata.json）。

---

### 4.3 用户查进度

**用户操作**：
> 任务跑得怎么样了？task_id=a3b8c0d1

**系统连锁反应**：

23. MCP Server 调 `tools.tool_check_status`。

24. 从 SQLite 拉 meta + 最近 15 条进度事件。

25. 返回类似：
> task_id=a3b8c0d1
> lifecycle=running
> stage=researcher (loop 2/3)
> graph_checkpoint: loop_count=2, card_count=47, final_report_ready=False
> 进度时间线（最近 5 条）：
>   2024-... section_done section_id=S2
>   2024-... reflector_done
>   2024-... section_start section_id=S3
>   ...

---

### 4.4 用户拿结果

**用户操作**：
> 给我看报告

**系统连锁反应**：

26. MCP Server 调 `tools.tool_get_result`。

27. 从 SQLite 读 meta（包含 report_path）。

28. 读 report.md 的前 1500 字符作为预览。

29. 返回路径 + 预览 + quality_review + section_conflicts（如果有）+ citation_audit。

30. 用户在客户端看到内容，可以直接打开本地 Markdown 文件查看完整报告。

---

### 4.5 用户追问

**用户操作**：
> 在这份报告基础上，再深挖一下 RISC-V 在 AI 推理加速场景的应用

**系统连锁反应**：

31. MCP Server 调 `tools.tool_follow_up_research`。

32. 从 checkpoint **恢复原任务的完整 state**（包括所有之前收集的卡片）。

33. 让 planner 分析这个追问：是"深挖某个已有 section"还是"加一个新 section"？

34. 生成新的 sub_tasks（不重做已有的工作）。

35. 启动一个**增量图执行**：dispatch_sections → researcher → ... → writer。

36. writer 拿到新旧卡片合在一起写**新版本的报告**，保存为 version 2。

37. 用户可以用 `compare_report_versions(task_id, 1, 2)` 看两个版本的差异。

---

这就是用户视角的完整链路。下面章节会深入各个节点的实现。

---

## 第 5 章 整体架构分层

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5：客户端                                                    │
│  Claude Desktop / Claude Code / Codex / Cursor / Nanobot           │
│  （MCP 客户端，通过 stdio 跟我们的 server 进程通信）                │
└─────────────────────────────────────────────────────────────────────┘
                                ▲
                                │ MCP 协议 (JSON-RPC over stdio)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4：MCP 工具层 (deep_research_mcp.py + tools.py)              │
│  - 7 个 MCP 工具，每个对应一个 tool_xxx 方法                        │
│  - 输入校验（长度、必填）                                           │
│  - 调用 Service 层                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                ▲
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 3：服务编排层 (service.py + runtime.py)                      │
│  - DeepResearchService：单例，串起所有组件                          │
│  - AgentContext：依赖注入容器（每个 agent 拿到自己需要的依赖）      │
│  - 4 个 Agent 实例（planner / researcher / reviewer / writer）      │
│  - Worker 子进程生命周期管理                                        │
│  - LangSmith 追踪配置                                               │
└─────────────────────────────────────────────────────────────────────┘
                                ▲
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 2：工作流层 (graph.py)                                       │
│  - LangGraph 编译后的 StateGraph                                    │
│  - 8 个节点 + 边定义（条件边、Send API）                            │
│  - 状态合并 reducer（_merge_sub_tasks / operator.add）              │
│  - AsyncSqliteSaver checkpoint（图状态持久化）                      │
└─────────────────────────────────────────────────────────────────────┘
                                ▲
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1：基础组件层                                                │
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐ │
│  │ SearchService       │  │ ModelWorker         │  │  Storage     │ │
│  │ (search_service.py) │  │ (worker.py 子进程)  │  │ (storage.py) │ │
│  │ - 12 种搜索引擎     │  │ - bge embedder      │  │ - aiosqlite  │ │
│  │ - 内容提取          │  │ - bge reranker      │  │ - WAL 模式   │ │
│  │ - 熔断 + 限流       │  │ - FAISS 检索        │  │ - 3 张表     │ │
│  └─────────────────────┘  └─────────────────────┘  └──────────────┘ │
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌──────────────┐ │
│  │ KnowledgeCache      │  │ QualityMixin        │  │  Retry/CB    │ │
│  │ (URL+claim 去重)    │  │ (quality.py)        │  │  Recency     │ │
│  │                     │  │ - 规则评分          │  │  QueryReform │ │
│  │                     │  │ - LLM 评审          │  │  Conflict    │ │
│  │                     │  │ - 大纲源筛选        │  │  Grounding   │ │
│  └─────────────────────┘  └─────────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                                ▲
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 0：外部依赖                                                  │
│  - LLM 提供商（DeepSeek/OpenAI/Anthropic/Gemini，via LiteLLM）      │
│  - 搜索 API（Tavily/Exa/Serper/Bocha/SemanticScholar/arXiv/...）    │
│  - 本地模型权重（bge-small-zh / bge-reranker-base）                 │
│  - SQLite 文件                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

每一层只依赖下一层，不反向依赖。

---

## 第 6 章 核心节点深入讲解

### 6.1 Supervisor 节点 ——「把策略变成可执行的待办」

#### 输入

- `topic`：用户问题
- `approved_plan`：用户确认的 markdown 策略（可能是用户改过的版本）
- `execution_plan`：上一步 draft 产出的结构化策略

#### 它做什么

```
当 Supervisor 被调用，系统首先会：

1. 检查是否已经有 execution_plan
   - 如果有（用户已确认），跳到步骤 3
   - 如果没有，调用 planner.draft_execution_plan 重新生成

2. 让 planner 把 execution_plan.query_strategy 中的每个研究方向
   展开成多个 SubTask
   - 例如方向 "Section 1: 芯片厂商进展" 展开成：
     - SubTask A: query="SiFive 2025 server chip"
     - SubTask B: query="Tenstorrent Wormhole 2025"
     - SubTask C: query="Ventana Veyron V2"

3. 让 planner 给每个 SubTask 加工：
   - 用规则推断 gap_type（这个查询是想填什么类型的信息缺口？
     definition / quant_support / counter_evidence / recent_update ...）
   - 用模板生成 rewritten_queries（同一个意图的 3-4 个不同表达）
   - 根据优先级配 search_profile（high 优先级 → 多搜几条结果）
   - 标注 required_source_types / required_evidence_types

4. 把所有 SubTask 写回 state，准备分发
```

#### 输出

- `sub_tasks`：待执行的所有研究子任务
- `execution_plan`：最终策略（如果是重新生成的）
- `cards_before_loop`、`previous_coverage`、`section_results`：清空初始值

#### 为什么这样做

- **分离"策略"和"执行"**：策略层（用什么搜索引擎、什么类型的源）是 LLM 决策的，但具体的 query 怎么写、retry 多少次是规则 + 模板的——这样 LLM 不需要重复想 query 模板，省 token
- **规则 + LLM 两层结合**：纯规则太死板（没法应对用户写的怪问题），纯 LLM 太贵（每次 query 改写都调一次 LLM 不划算）

---

### 6.2 Section Researcher 节点 ——「真正干活的人」

这是整个系统最复杂的节点之一，是 LangGraph 通过 Send API **并行运行多个实例**的节点。

#### 输入（每个实例独立）

`SectionResearchInput`：
- task_id, topic
- section_id, section_title
- pending_tasks（属于这个 section 的所有 SubTask 列表）
- existing_card_count（用来给新卡片分配 unit_id）

#### 它做什么

```
当一个 section_researcher 实例被启动，系统首先会：

1. 给这个 section 单独建一个 KnowledgeCache 实例
   - 用于本 section 内的去重（URL hash + claim hash）

2. 对该 section 的每个 pending_task 顺序执行（同一 section
   内的 task 不并行，避免抢占同一搜索引擎的并发额度）：

   2.1 【Tier 1：带退避 + 抖动的并发搜索】
       - 用 task.rewritten_queries 同时去多个搜索引擎并发搜
       - 每个 retriever 调用都经过熔断器和限流器（详见第 8 章）
       - 失败 → 指数退避（带 jitter） → 最多重试 N 次
       - 如果全部返回 0 结果，进入 Tier 2

   2.2 【Tier 2：多策略查询重写】（如果 Tier 1 一无所获）
       - 单次 LLM 调用，要求生成 3 个不同策略的备选 query
         （SIMPLIFY / SYNONYMS / DECOMPOSE / BROADEN_TIME / SPECIFY）
       - 顺序试这 3 个 query，第一个有结果就停
       - 详见第 11 章亮点②

   2.3 【Tier 3：优雅降级】（Tier 1 + 2 都失败）
       - 不抛异常、不阻塞其他 task
       - 把 SubTask 标记 status=degraded
       - degradation_reason 写入
       - 跳到下一个 task

   2.4 【打分排序 + 去重】
       - 对拿到的所有文档算 search_quality_score
         （内容长度、来源类型加分、对时效任务做 recency 衰减）
       - 按 URL 去重，每个 URL 留分数最高的那一版
       - 详见第 11 章亮点③（recency 衰减）

   2.5 【调用 model worker 做 embedding + 重排序】
       - 把所有文档发给独立子进程
       - 子进程把文档切句、嵌入、用 FAISS 召回 top-K、再用 cross-encoder 精排
       - 返回最相关的句子级 evidence 列表

   2.6 【LLM 抽取 KnowledgeCard】
       - 把 evidence 喂给 LLM，prompt 要求输出 JSON 卡片列表
       - 每张卡片有 claim、evidence_summary、exact_excerpt、
         confidence、stance、claim_type、time_scope、entities
       - 用 evidence_ids 锚定卡片到具体证据，防止编造

   2.7 【KnowledgeCache 去重】
       - 检查每张新卡片：URL + claim 的 MD5 hash 已经存在吗？
       - 存在 → 丢弃
       - 不存在 → 加入 cache，加入返回列表

3. 把所有卡片 + 任务状态返回给图状态

输出：
- knowledge_cards: 本 section 的新卡片
- sub_tasks: 更新过状态的 SubTask 列表
- section_results: 本 section 的统计（success/failed/degraded 数）
```

#### 为什么这样设计

- **3 层错误恢复**确保**局部失败不影响整体**：单个搜索引擎抽风不会让整个任务失败
- **每 section 一个 KnowledgeCache** + 最后全局再去重 → 避免本 section 重复处理同一个 URL，**最终又能跨 section 去重**
- **LLM 抽取卡片用 evidence_ids 锚定**：LLM 不能凭空写"我觉得这篇文章说了 X"，必须指明是 evidence-id-7 里的句子说的

---

### 6.3 Reflector 节点 ——「质检员 + 决策者」

#### 输入

- 全部 knowledge_cards
- execution_plan
- 之前的 sub_tasks（含已完成的）
- loop_count（当前是第几轮反思）
- previous_coverage、cards_before_loop（上一轮的状态，用来算"本轮提升"）

#### 它做什么

```
当 Reflector 被调用，系统首先会：

1. 【按 section 评审】
   - 对每个 section_id 独立评审
   - 评审分两层：

   1.1 规则评审（rule_based_section_review）：
       - 覆盖度 = 已被卡片回答的子问题 / 总子问题数
       - 证据数得分 = min(1.0, 卡片数 / 4)
       - 来源多样性 = min(1.0, 不同域名数 / 3)
       - 主要来源占比 = primary_source 卡片数 / 总数
       - is_enough 判断：覆盖 ≥ 0.7 AND 卡片 ≥ 3 AND 域名 ≥ 2 AND 主要来源 ≥ 0.25

   1.2 LLM 评审（llm_section_review）：
       - 让 LLM 看 section 摘要 + top 卡片，判断
         "语义上是否足够回答这个 section 的核心问题"
       - 输出 semantic_coverage_score、support_score、conflict_score
       - 列出 missing_questions 和 weak_claims

   1.3 合并：
       - 如果 LLM 说"不够" → 强制把 is_enough 改 False
       - 这是 LLM 否决权（防止规则评审过于宽容）

2. 【构建 section_digests】
   - 每个 section 一个 SectionDigest，是该 section 卡片的压缩版
   - 包含 key_claims、items（精选卡片）、coverage_score 等
   - 这个 digest 后面会喂给 outline_builder 和 writer，
     **替代海量原始卡片**避免 LLM 上下文爆炸

3. 【算饱和度】（compute_saturation_score）
   - coverage_delta = 当前 coverage - previous_coverage
   - marginal_gain = 新增卡片数 / 总卡片数
   - saturation = 1.0 - (coverage_delta * 0.6 + marginal_gain * 0.4)
   - 物理意义：如果本轮没带来啥覆盖度提升、新卡片占比也很低
     → 饱和度高 → 该停了

4. 【决定下一步走向】（should_stop_early）
   - 检查 4 个条件：
     a) loop_count >= 最小轮数（默认 1）？  (没到，必须继续)
     b) saturation_score >= 阈值（默认 0.85）？  (够饱和，停)
     c) 还有未满足的 follow_up_requests 吗？  (没了，停)
     d) loop_count >= 最大轮数（默认 3）？  (硬上限，强停)
   - 综合决定：
     - 停 → route_to = "outline_builder"
     - 不停 → route_to = "dispatch_sections"，生成补研究 sub_tasks

5. 【生成补研究任务】（如果不停）
   - 对每个 is_enough=False 的 section：
     - 把它的 missing_questions、gap_types、required_source_types
       打包成新的 SubTask
   - 让 planner.prepare_subtasks_for_search 给这些新 task 加工
     （生成 rewritten_queries、search_profile 等）
   - 这些 task append 到 sub_tasks，会被下一轮 dispatch 派出去

6. 【评估降级任务影响】（assess_degraded_impact）
   - 找出 status=degraded 的 task 都在哪些 section
   - 标记 critical_gaps（高优先级 section 受影响）
   - 写入 quality_review，让 writer 阶段在报告里加 caveat

输出：
- route_to: "outline_builder" 或 "dispatch_sections"
- section_digests
- quality_review（综合质量摘要）
- saturation_score, previous_coverage（给下一轮用）
- 可能的新 sub_tasks
```

#### 为什么这样设计

- **规则评审 + LLM 评审 + 否决权**：单层都不够鲁棒。规则评审会被"卡片很多但都是低质量重复"骗过；LLM 评审有时候过于宽容。两层结合避免单点失效。
- **饱和度自适应早停**：固定循环数（比如永远跑 3 轮）会浪费——简单问题 1 轮就够。饱和度公式 = 60% 看覆盖度提升 + 40% 看边际增益，让系统**该停就停**。
- **section_digest 取代原始卡片送入下游**：原始卡片 50+ 张总长可能 50000 字符，喂给 outline_builder 直接上下文爆炸。digest 压缩到几千字符，**保留判断需要的信息**（key_claims、是否够用、缺什么）。

---

### 6.4 Outline Builder 节点 ——「证据驱动的大纲生成」

#### 输入

- section_digests
- execution_plan（最初的策略）
- 全部 knowledge_cards（备用，主要用 digest）

#### 它做什么

```
当 Outline Builder 被调用，系统首先会：

1. 把 section_digests 序列化成 LLM prompt 中的"已有证据包"
   - 每个 digest 显示 section_id、title、key_claims、coverage_score、items（前几条）

2. 让 LLM 基于这些证据，输出最终报告的章节大纲：
   - report_title（报告标题）
   - user_goal（用户目标的精炼描述）
   - sections（章节列表，每个章节有 section_id、title、purpose、
     evidence_digest_ids（用到哪些证据 digest）、evidence_requirements
     （这个章节需要哪类证据））
   - outline_notes（生成大纲时的备注）

3. 如果 LLM 失败，回退到 fallback 策略：
   - 把每个 research track 直接当一个章节（research_tracks_as_sections）

4. 把 outline 和 plan_data 写入 state

输出：
- evidence_outline: 大纲对象
- plan_data: 给 writer 直接用的简化版（report_title + sections + ...）
```

#### 为什么这样设计

- **大纲在证据收集**之后**才生成**：直接在 draft 阶段写大纲，会发现"哎呀我想讲 X 但没找到资料"。后置大纲 → 大纲一定能配上证据
- **digest 而非原始卡片**：上下文友好，且避免 LLM 被某一篇文章的具体表述带偏

---

### 6.5 Detect Conflicts 节点 ——「跨源分歧主动暴露」

详见第 11 章亮点④。

---

### 6.6 Writer 节点 ——「结构化 grounded 报告生成」

详见第 11 章亮点④（这是最复杂的亮点）。

---

## 第 7 章 数据中间层 ——「从网页到报告，证据怎么流动」

这一章不讲节点，讲数据。从用户输入一个 query 开始，数据形态如何一步步演变：

```
┌─────────────────────────┐
│  用户原始 query         │
│  "RISC-V 数据中心"      │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Seed Queries (planner) │
│  ["RISC-V data center", │
│   "RISC-V server 2025", │
│   ...]                  │
└───────────┬─────────────┘
            ▼ Tavily/Exa/...
┌─────────────────────────────────────────────────────────────────┐
│  Raw Search Results (来自不同 retriever，格式不一)              │
│  Tavily: {url, title, raw_content, published_date, score}       │
│  Exa: {url, title, text, publishedDate, score}                  │
│  SemanticScholar: {title, abstract, year, citations, authors}   │
└───────────┬─────────────────────────────────────────────────────┘
            ▼ normalize_document (统一字段)
┌─────────────────────────────────────────────────────────────────┐
│  Document (标准化文档)                                          │
│  {                                                              │
│    document_id, url, title, content, raw_content,               │
│    source_name (tavily/exa/...), source_layer (general/vertical),│
│    source_kind (web/paper/pdf/repo/news/...),                   │
│    published_time, year, authors, venue, doi, pdf_url,          │
│    score (含 recency 衰减),                                     │
│    search_quality_score                                         │
│  }                                                              │
└───────────┬─────────────────────────────────────────────────────┘
            ▼ model worker: 切片 + 嵌入 + FAISS + reranker
┌─────────────────────────────────────────────────────────────────┐
│  Evidence (句子级证据，从 worker 返回)                          │
│  [{                                                             │
│    score, url, title, published_time, page_type,                │
│    source_type, content_quality_score,                          │
│    excerpt (单个父 chunk 的全文)                                │
│  }, ...]                                                        │
└───────────┬─────────────────────────────────────────────────────┘
            ▼ LLM 抽取
┌─────────────────────────────────────────────────────────────────┐
│  KnowledgeCard (知识卡片，业务上的最小单元)                     │
│  {                                                              │
│    unit_id: "U001",                                             │
│    section_id: "S2",                                            │
│    claim: "SiFive Performance P870 在数据中心场景表现优于...",  │
│    evidence_summary: "SiFive 公布的 benchmark 数据显示...",     │
│    exact_excerpt: "P870 achieves 95% of Cortex-A78's IPC...",   │
│    source: "https://sifive.com/...",                            │
│    source_title: "SiFive Announces P870",                       │
│    source_type: "primary_source",                               │
│    confidence: "high",                                          │
│    stance: "supporting",                                        │
│    claim_type: "metric",                                        │
│    time_scope: "recent",                                        │
│    entities: ["SiFive", "P870", "Cortex-A78"],                  │
│    evidence_strength: "strong",                                 │
│    evidence_score: 0.87                                         │
│  }                                                              │
└───────────┬─────────────────────────────────────────────────────┘
            ▼ KnowledgeCache 全局去重
            ▼ Reflector 按 section 整理
┌─────────────────────────────────────────────────────────────────┐
│  SectionDigest (章节证据包，给 writer 用的压缩版)               │
│  {                                                              │
│    section_id, title, purpose, questions,                       │
│    coverage_score, evidence_count_score,                        │
│    source_diversity_score, is_enough,                           │
│    review_reason, missing_questions, key_claims,                │
│    items: [SectionDigestItem, ...]  (精选的几张卡片)            │
│  }                                                              │
└───────────┬─────────────────────────────────────────────────────┘
            ▼ Outline Builder 基于 digest 生成大纲
┌─────────────────────────────────────────────────────────────────┐
│  EvidenceOutline (最终报告大纲)                                 │
│  {                                                              │
│    report_title, user_goal,                                     │
│    sections: [{section_id, title, purpose,                      │
│                 evidence_digest_ids, evidence_requirements},...]│
│    outline_notes                                                │
│  }                                                              │
└───────────┬─────────────────────────────────────────────────────┘
            ▼ Detect Conflicts 加冲突信息
            ▼ Writer 按 section 写 + 渲染引用
┌─────────────────────────────────────────────────────────────────┐
│  Final Report                                                   │
│  (Markdown 文件 + cards.json + sources.json + metadata.json     │
│   + citation_audit + section_conflicts)                         │
└─────────────────────────────────────────────────────────────────┘
```

### 每一层为什么必须存在

| 层 | 没这层会怎么样 | 这层解决什么 |
|---|---|---|
| Raw → Document | 各 retriever 字段名不一样，下游要写 N 套适配 | 统一字段名，下游只认 Document 一种 |
| Document → Evidence | 整篇文章 5000 字直接喂 LLM 上下文爆炸 | 切句 + 嵌入 + 排序，只留最相关的几十句 |
| Evidence → KnowledgeCard | 句子还是太碎，没有"论点"概念 | 抽出 claim + evidence_summary，结构化 |
| KnowledgeCard → SectionDigest | 50+ 张卡片喂 writer 还是太多 | 压缩到 5-10 张代表性卡片 + 摘要分 |
| SectionDigest → Outline | 没有大纲就没有报告骨架 | 基于真实证据决定章节怎么排 |
| Outline → Report | 输出层 | 最终交付物 |

---

## 第 8 章 双层检索体系

### 8.1 架构

```
┌────────────────────────────────────────────────────────────────────┐
│  用户 query: "RISC-V 数据中心"                                     │
└────────────────────────────────┬───────────────────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  SearchService.search() │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌──────────┐      ┌─────────────┐    ┌──────────────┐
        │ 推断垂直 │      │ 通用层      │    │ 垂直层       │
        │ 领域     │      │ (general)   │    │ (vertical)   │
        │          │      │             │    │              │
        │"academic"│      │ Tavily      │    │ SemScholar   │
        │"papers"等│      │ Exa         │    │ arXiv        │
        │ 关键词   │      │ Serper      │    │ PubMed       │
        │ 触发学术 │      │ Bocha       │    │ ...          │
        │          │      │ DuckDuckGo  │    │              │
        │          │      │ SerpAPI/Bing│    │              │
        │          │      │ Google/Searx│    │              │
        └──────────┘      └──────┬──────┘    └───────┬──────┘
                                 │                   │
                                 ▼                   ▼
                    每个 retriever 都经过：
                    ┌─────────────────────────────────────┐
                    │  per-engine 熔断器（CircuitBreaker） │
                    │  per-engine 限流器（Semaphore）     │
                    └─────────────┬───────────────────────┘
                                  ▼
                         并发执行（asyncio.gather）
                                  ▼
                       Document[] 合并 + 去重 + 排序
                                  ▼
                       内容补全（_extract_missing_content）
                       Jina Reader / Readability / PyMuPDF
                                  ▼
                            最终 Document[]
```

### 8.2 通用层（General）的 9 个 retriever

| 名字 | 用什么 API | 特点 | 主用途 |
|---|---|---|---|
| Tavily | api.tavily.com | AI 优化，返回正文 | **默认主力** |
| Exa | api.exa.ai | 神经网络语义搜索 | 技术/长文 |
| Serper | google.serper.dev | Google 包装 | 新闻类 |
| Bocha | api.bochaai.com | 中文语义 | 中文场景 |
| SerpAPI | serpapi.com | Google 包装 | 备选 Google |
| Bing | api.bing.microsoft.com | 必应 | 微软生态 |
| Google | googleapis.com | Custom Search | 自有 CSE |
| Searx | 自建 | 元搜索 | 自部署 |
| DuckDuckGo | ddgs 库 | 免费无 key | **兜底** |

### 8.3 垂直层（Vertical）的 3 个 retriever

| 名字 | 用什么 API | 内容类型 |
|---|---|---|
| SemanticScholar | api.semanticscholar.org | 学术论文 + 引用数 |
| ArXiv | export.arxiv.org | 物理/CS 预印本 |
| PubMedCentral | eutils.ncbi.nlm.nih.gov | 生物医学论文 |

**触发逻辑**：`infer_verticals()` 看 query 里有没有 "paper" / "journal" / "arxiv" / "论文" / "期刊" / "学术" 等关键词，有则触发学术垂直层。

### 8.4 熔断器（CircuitBreaker）

**问题**：某个搜索 API 抽风（5xx、超时）时，每次研究任务都会去试一下，浪费时间。

**机制**：
- 每个 retriever 一个独立的 breaker
- 连续失败 N 次（默认 3） → 标记 OPEN
- OPEN 状态下，cooldown 时间内（默认 60s）直接返回空列表，**不发请求**
- cooldown 结束自动恢复（隐式 close）
- 成功一次就重置失败计数

### 8.5 内容补全

很多搜索 API 只返回标题 + 摘要，正文还得自己抓。系统按以下顺序尝试：

```
拿到 Document，content 长度 < 500 → 需要补内容
  │
  ▼
URL 是 PDF？
  ├─ 是 → 用 PyMuPDF 抓 PDF 文本
  └─ 否 → 走网页抓取
            │
            ▼
       配了 Jina API key 或允许免 key 用 Jina？
       ├─ 是 → 调 Jina Reader（处理 JS 渲染、反爬）
       └─ 否 → Readability 本地解析（lxml + readability-lxml）
                └─ 失败 → 用 lxml XPath 兜底，抓 h1-h3/p/li/blockquote/pre/table
```

最终 content 限制在 20000 字符以内（防止单篇文章吃掉 context）。

---

## 第 9 章 Model Worker 子进程

### 9.1 为什么要独立进程

bge-small-zh embedder 加载需要 5-10 秒，bge-reranker 加载需要 3-5 秒，加载时 Python 主进程**完全卡住**，连不上 LangGraph 也卡住。

**方案**：把这两个模型放到独立子进程里。主进程只通过队列发任务、收结果。

### 9.2 通信结构

```
┌──────────────────────────────────────────────────────────────────┐
│  主进程 (DeepResearchService)                                    │
│                                                                  │
│  ┌──────────────────────────────────────────────┐                │
│  │  asyncio 事件循环                            │                │
│  │  - 处理 MCP 请求                             │                │
│  │  - 运行 LangGraph                            │                │
│  │  - 跟外部 LLM / 搜索 API 通信                │                │
│  └──────────────────────────────────────────────┘                │
│                                                                  │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────┐             │
│  │ job_queue  │  │result_queue│  │ worker_state    │             │
│  │ (mp.Queue, │  │ (mp.Queue) │  │ (mp.Manager     │             │
│  │ maxsize=64)│  │            │  │  shared dict)   │             │
│  └────────────┘  └────────────┘  └─────────────────┘             │
└──────────┬─────────────────────────────────┬──────────┬──────────┘
           │ put job (async)        get result│         │ poll
           │                                  │         │
           ▼                                  ▲         ▼
┌──────────────────────────────────────────────────────────────────┐
│  子进程 (model_worker_process)                                   │
│                                                                  │
│  1. 启动时按顺序加载：                                           │
│     transformers → torch → sentence-transformers (bge embedder)  │
│     → CrossEncoder (bge reranker) → faiss                        │
│     每一步把 worker_state["status"] 改成 "LOADING_*" 让主进程看到│
│                                                                  │
│  2. 心跳线程：每 5 秒更新 worker_state["heartbeat"] = now()       │
│     主进程靠这个判断子进程是不是卡死了                           │
│                                                                  │
│  3. 主循环：                                                     │
│     while True:                                                  │
│       task = job_queue.get()                                     │
│       worker_state["status"] = "BUSY"                            │
│       result = process_task(task)                                │
│       result_queue.put(result)                                   │
│       worker_state["status"] = "READY"                           │
└──────────────────────────────────────────────────────────────────┘
```

### 9.3 一个 job 怎么被处理

主进程往 job_queue 发一个 job：
```
{
  "job_id": "abc123",
  "queries": ["RISC-V 数据中心", "RISC-V server adoption"],
  "documents": [Document, Document, ...]  // 比如 50 篇网页
}
```

子进程处理流程：

```
1. 切片（chunk）：对每篇文档：
   - 按段落切（paragraph）
   - 太长的段落进一步按句子切
   - 每个 chunk 大小约 800 字符，相邻 chunk 有 150 字符重叠
   - 给每个 chunk 算一个 summary（摘要）
   生成"父 chunk"列表（用于最终 evidence 输出）

2. 提取句子（子 chunk）：把每个父 chunk 切成句子
   - 用正则 `[。！？.!?]` 切
   - 过滤掉 < 15 字符的句子
   - 给每个句子构造嵌入文本：
     "title: ... | page_type: ... | source_type: ... | domain: ... |
      parent_summary: ... | sentence: ..."

3. 嵌入（embedding）：用 bge embedder 一次把所有句子嵌入成向量
   （normalized）

4. 建 FAISS 索引：内积索引（normalized → 内积 = 余弦相似度）

5. 召回（recall）：对每个 query：
   - 嵌入这个 query
   - 在 FAISS 里找 top (8 × max_results) 个最相似的子 chunk
   - 记下它们对应的父 chunk 索引

6. 重排（rerank）：对每个 (query, 父 chunk) 对：
   - 构造重排文本：
     "title: ... | page_type: ... | source_type: ... |
      summary: ... | content: 父 chunk 全文"
   - 用 bge cross-encoder 算 (query, 文本) 的相关性分

7. 去重 + 排序：按分数降序，去重 (url, excerpt)，
   保留 top (search_max_results + 1)

8. 包装成 evidence 列表返回到 result_queue
```

### 9.4 子进程的可靠性保障

- **心跳检测**：主进程发现 heartbeat 超过 300 秒没更新 → 认为子进程卡死 → kill + 重启
- **重启预算**：5 分钟内最多重启 5 次（滑动窗口），超额拒绝再启动，避免无限重启循环
- **背压（backpressure）**：job_queue 有 maxsize=64，满了之后 put 会阻塞 → 自然减速主进程提交速率，避免 OOM

---

## 第 10 章 持久化与可靠性

### 10.1 三张 SQLite 表

```
┌──────────────────────────────────────────────────────────────┐
│  TaskRegistry SQLite (~/Desktop/DeepResearch/_runtime/)      │
│  开启 WAL 模式 (PRAGMA journal_mode = WAL)                   │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │ tasks                                            │        │
│  │ - task_id (PK)                                   │        │
│  │ - meta_json (任务元数据 JSON, 包括 lifecycle)    │        │
│  │ - draft_json (草拟策略 JSON)                     │        │
│  │ - status_text (人类可读的当前状态)               │        │
│  │ - created_at, updated_at                         │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │ task_events (索引 task_id + id)                  │        │
│  │ - id (AUTOINCREMENT)                             │        │
│  │ - task_id                                        │        │
│  │ - timestamp, stage, level, message               │        │
│  │ - extra_json (额外字段，stage='progress' 时存    │        │
│  │   结构化进度事件)                                │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │ report_versions (UNIQUE(task_id, version))       │        │
│  │ - id, task_id, version, content                  │        │
│  │ - change_summary, created_at                     │        │
│  │ 每次 follow_up 产出新版本                        │        │
│  └──────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  LangGraph Checkpoint SQLite (独立文件)                      │
│  由 AsyncSqliteSaver 管理                                    │
│  - 每个节点执行完之后写入一次 state 快照                     │
│  - 用 thread_id (= task_id) 关联同一任务的多次快照           │
│  - 支持任务中断恢复 + follow_up 增量研究                     │
└──────────────────────────────────────────────────────────────┘
```

### 10.2 WAL 为什么重要

没开 WAL 时：读和写互相阻塞。一边在写卡片，一边查状态的请求要等。

开了 WAL 后：读和写可以同时进行，只有"多个写"才会串行。本项目典型场景是：
- 1 个写者（researcher 在 append events）
- N 个读者（status 工具在被 N 个客户端查询）

WAL 大幅减少这种场景的延迟。

### 10.3 异步 SQLite

用 `aiosqlite` 替代同步 `sqlite3`。差别：
- 同步 `sqlite3.execute()` 会阻塞调用它的协程
- `aiosqlite.execute()` 把实际 SQL 放到一个 IO worker 线程跑，await 即可，主事件循环不阻塞

### 10.4 失败时持久化完整 traceback

任务失败时（异常抛出），系统会：
- 截取完整 Python traceback（限制 16KB，超过取头尾各 8KB 中间省略）
- 写入 `task_events.extra_json`
- 写入 `tasks.meta_json` 的 `exception_type` 字段
- 通过 probe 文件落盘

这样事后 `get_research_status` 能拿到完整失败信息，不用复现。

### 10.5 历史数据维护

CLI 命令 `deep-research-mcp prune` 提供清理：
- 默认保留 30 天内的终态任务（completed / failed / cancelled）
- 30 天前的任务的 checkpoint 行 + 注册表行一起删
- 自动发现 LangGraph 的 checkpoint 表（用 `PRAGMA table_info` 找含 `thread_id` 列的表），不写死表名
- 支持 `--dry-run` 预览将删除什么
- VACUUM 收缩文件（Windows 上有锁会跳过，best-effort）

---

## 第 11 章 亮点设计（每个都按 ① 干嘛 ② 不做会怎样 ③ 怎么做 ④ 为什么这样）

这一章是面试和写简历的重点。每个亮点单独讲完。

---

### 亮点① Reflector 自适应早停（基于语义饱和度）

**① 这是干嘛的**

让系统在"已经查够了"的时候**自动停下**，不要傻乎乎地把循环数跑满。

**② 不做会怎样**

只能用固定循环数（比如永远跑 3 轮）：
- 简单问题：1 轮就够，多跑 2 轮浪费 LLM 调用 + 搜索 API 配额
- 难问题：3 轮还不够，但被硬上限卡住，报告残缺

**③ 怎么做（流程级）**

```
每轮 reflector 结束时计算：

1. coverage_delta = 当前 coverage - 上一轮 coverage
   含义："本轮覆盖度提升了多少"

2. marginal_gain = 本轮新增卡片 / 总卡片数
   含义："本轮新增证据占比"

3. saturation = 1.0 - (coverage_delta * 0.6 + marginal_gain * 0.4)
   含义：如果本轮提升 + 新增都很小 → 饱和度高 → 该停

4. 决定是否停：
   - 没到最小轮数 → 必须继续
   - 没未满足缺口了 → 停
   - 饱和度 >= 0.85 → 停
   - 达到最大轮数 → 强停（safety net）
```

**④ 为什么这样**

- **权重 60/40**：覆盖度提升比新增数量更重要。你新增 30 张卡片但都没回答新问题，没用。
- **小阈值 0.85 而非 0.95**：deep research 任务允许少量未覆盖问题（永远完美不现实），0.85 是经验值。
- **保留最大轮数 safety net**：饱和度算错的情况下，不能让任务无限循环。

---

### 亮点② 多策略查询重写（1 → 3，单 LLM 调用）

**① 这是干嘛的**

搜索拿到 0 结果时，自动换 3 种不同思路重写 query，提升召回率。

**② 不做会怎样**

旧版本：0 结果 → 1 次 LLM 简化 → 还 0 → 放弃。

实际遇到的失败模式有多种：
- query 太长太具体 → 应该简化（旧版本能处理）
- 用了错误术语 → 应该换同义词（旧版本不行）
- query 是复合问题 → 应该拆开（旧版本不行）
- 加了时间限定但没那么新的资料 → 应该去掉时间词（旧版本不行）

旧版本只覆盖第一种情况。

**③ 怎么做（流程级）**

```
当 researcher 发现某个 query 返回 0 结果：

1. 单次 LLM 调用，prompt 内显式枚举 3 种策略，要求每种各写一个备选：
   - SIMPLIFY: 去定语、更泛化
   - SYNONYMS: 关键术语换同义/中英版本
   - DECOMPOSE: 拆成 2 个子问题

2. LLM 返回 JSON: {"queries": ["alt1", "alt2", "alt3"]}

3. researcher 顺序试这 3 个 query：
   - 第一个有结果 → 用它，记录 result["reformulated_query"] = "alt1"
   - 全部失败 → 进入 Tier 3 降级

4. LLM 调用次数：固定 1 次（不管尝试几个备选）
```

**④ 为什么这样**

- **单次 LLM 调用而非 3 次**：3 次串行 LLM 太贵且太慢。1 次让 LLM 一口气写 3 个备选，成本和原来一样。
- **明确枚举策略**：让 LLM 直接随便写 3 个备选，它倾向于写 3 个相似的（同义改写）。**用名字 + 描述强制策略多样性**。
- **顺序试而非并发试**：并发会浪费——如果第一个就成功，后两个就白搜了。

---

### 亮点③ 时效衰减打分（half-life 模型）

**① 这是干嘛的**

时效敏感任务（"2025 最新"类）下，老资料的排名自动降低。

**② 不做会怎样**

打分公式只看"内容长度 + 来源类型"，2019 年的综述论文（内容长）会击败 2025 年的预印本（内容相对短）。用户看到结果里全是过时的内容。

**③ 怎么做（流程级）**

```
当 researcher 给 Document 打分时：

1. 基础分 = provider_score + 内容长度bonus + 垂直层bonus + 文档类型bonus
   （原来的公式不变）

2. 如果开启了 recency_weighting 且 task 有 time_scope：
   - 提取 Document 的 published_time（支持多种格式：
     "2024-03-15"、"2024"、"Mar 15, 2024"、ISO 8601...）
   - 选 half_life：
     · time_scope = "recent" → 6 个月
     · time_scope = "current" / "current_year" → 18 个月
     · time_scope = "future" → 12 个月
     · 其他 → 36 个月默认（很温和的衰减）
   - 算 age_months（未来日期 clamp 到 0）
   - weight = 0.5 ^ (age_months / half_life)
   - 最终 weight floor 在 0.3（极旧也不到 0）
   - 最终分数 = 基础分 * weight

3. 把 weight 也存到 document["recency_weight"]，方便事后审计
```

**④ 为什么这样**

- **half-life 而非线性衰减**：信息陈旧速度本质上是指数衰减的（昨天的新闻比一周前的有用 50 倍，但一周前比两周前只有用 2 倍）
- **不同 scope 不同半衰期**：硬件话题（如 RISC-V）3 年内的内容都还行；新闻类 6 个月就过时
- **floor 在 0.3**：极旧的经典综述（比如 2010 年深度学习总结）还是有参考价值，不能完全抹掉
- **未知日期 → weight=1.0**：很多源没有 publish date 元数据，不能因此惩罚（那是 metadata 质量问题，不是 recency 问题）

---

### 亮点④ Grounded Citation Generation（写时强制结构化引用）

**这是最重要的亮点**，单独讲细一点。

**① 这是干嘛的**

让 writer 写出来的报告**物理上无法编造引用号**，所有 `[3]` 都一定对应真实存在的来源 #3，且引用的内容跟来源的实际文字对得上。

**② 不做会怎样**

旧版本（事后审计模式）：
- LLM 自由写 markdown，里面带 `[1][2]`
- 系统事后扫一遍，看 `[1]` 是不是在 allowed_urls 里
- 但是**没法检查"[1] 引用的句子是不是源 #1 真的说的"**

典型幻觉：LLM 写「根据 [3]，模型 X 准确率 95%」，但源 #3 实际写的是 87%。事后审计扫不出来。

这是 deep research 类产品**最大的信任问题**——报告看起来很专业，实际数据是飘的。

**③ 怎么做（流程级）**

```
当 writer 处理某一个 section：

1. 选出该 section 的 raw_cards（最多几张精选）

2. 给每张卡片打上短 ID：E1, E2, E3, ... EN
   同时记录每个 ID 对应的 reference_number（最终在报告里显示的引用号）

3. 构造 prompt 喂给 LLM：

   "你只能引用以下 evidence ID（不能编造其他 ID）：

   [E1] (ref 3)
     source: SiFive 官方
     excerpt: 'P870 achieves 95% of Cortex-A78's IPC...'

   [E2] (ref 4)
     source: Anandtech 评测
     excerpt: '...measured 87% in our internal tests...'

   规则：
   - 输出 JSON: {paragraphs: [{text, evidence_ids, quote?}]}
   - evidence_ids 只能从 [E1, E2, ..., EN] 选
   - 数字/百分比/日期/人名必须 quote 字段给出 verbatim 引文
   - text 里不要写 [1] 这种标记（系统自己渲染）"

4. LLM 返回 JSON

5. 系统验证：
   - parse JSON → 如果不是合法 JSON → 降级到老路径，记 fallback=True
   - 对每个段落：
     a) evidence_ids 检查闭集：不在 {E1..EN} 里的丢弃，记入 invalid_ids_dropped
     b) 如果有 quote 字段：quote 的 token 必须全部在引用的 excerpt 的 token 里
        （token 级 containment，容忍标点和空白差异）
        不通过 → 记入 quote_failures（不删段落，只标记）
     c) 段落文字里有数字 → 这些数字必须在引用的 excerpt 里有 ±5% 容差的匹配
        不通过 → 记入 numeric_failures（不删段落，只标记）
   - 如果所有段落 evidence_ids 全失效 → 整个 section 降级到老路径

6. 系统渲染 Markdown：
   - 段落文本 + 末尾 [3][4]
   - LLM 写的 [99] 之类的字符串被剥离（系统才有引用权）
   - [3][4] 这种纯文本最后由 _replace_numbered_citations_with_links
     转换成 <sup><a href="url">[3]</a></sup>

7. 累计 citation_audit：
   - sections_grounded / sections_fallback
   - citations_total / invalid_ids_dropped
   - quote_failures / numeric_failures
   - per_section 详情

8. 用户在 get_research_result 里看到这份 audit，可以一眼看出哪些
   引用可能漂移
```

**④ 为什么这样**

- **结构化 grounding 不是事后审计**：审计能发现"引用号无效"，但发现不了"引用号有效、内容飘的"。结构化在生成端拒绝问题。

- **evidence ID 不是 reference number**：直接告诉 LLM "你只能引用 [3][4]" 时，LLM 还是会"觉得"可以编 `[7]`。改用抽象 ID（E1, E2）让 LLM **只能从给定集合里选**，从字面上就少了编造冲动。

- **verbatim quote 校验是 token containment 而非精确子串**：
  - 精确子串：要求 quote 是 excerpt 的一段连续字符 → LLM 加个标点就过不了 → 假阳性高
  - token containment：quote 的 token 集合 ⊆ excerpt 的 token 集合 → 容忍标点/空白/顺序差异
  - 但要求所有 token 都在 → 不会被随手编一个的"看起来像引文"骗过

- **数值容差 5%**：
  - 太严（0%）：95% vs 95.3% 误报
  - 太松（20%）：95% vs 80% 漏报
  - 5% 是经验值——同一数据的不同小数舍入会过，真错会拦下

- **不阻塞、只标记 + 降级**：哪怕所有段落都有 numeric_failures，依然产出报告，只是 audit 里数字难看。**研究工具的核心是出报告，把问题暴露给用户决策，而不是自己 block**。

- **fallback 机制确保鲁棒性**：JSON 解析失败 / 所有 ID 失效 → 自动走老的 free-form 路径，老路径的输出样式跟 grounded 路径一致（都是超链接小角标）。**用户体验不变**，audit 里标记 writer_fallback=True 即可。

**关键技术对照**：

| | 旧（事后审计） | 新（写时 grounded） |
|---|---|---|
| LLM 能不能编引用号 | 能，事后系统替换为 plain text | **不能**，物理上 LLM 不写 `[N]` |
| 数字幻觉 | 检测不到 | 强制 verbatim quote + ±5% 数值校验 |
| 额外 LLM 调用 | +1-2 次 | 0 |
| 鲁棒性 | 单点 | JSON 解析失败自动降级 |
| 输出样式 | 小角标 + 末尾链接 | **一致**（小角标 + 末尾链接） |

---

### 亮点⑤ 跨源冲突检测

**① 这是干嘛的**

同一 section 内的卡片如果互相矛盾（A 说 95%，B 说 87%），让 writer 在报告里**显式写出"来源存在分歧"**，而不是随便挑一个。

**② 不做会怎样**

writer 看到两个相互矛盾的卡片，会按 prompt 倾向（"写得自信、给出明确判断"）随便选一个。读者根本不知道存在分歧。

**③ 怎么做（流程级）**

```
当 detect_conflicts 节点执行：

1. 按 section_id 分组卡片

2. 对每个 section，跳过卡片少于 min_cards（默认 3）的

3. 6 层过滤生成候选对：
   层 1: 全部两两组合
   层 2: 同 source_url 的对去掉（一篇文章不会自己矛盾）
   层 3: time_scope 不兼容的去掉（historical vs recent 不是矛盾，是时间序列）
   层 4: stance 检查：
         - supporting × counter 对 → 保留
         - 或 同实体 + 数值差异 > 2% → 保留为数值冲突候选
         - 其他 → 去掉（neutral / limitation 不算分歧）
   层 5: claim 文本相似度（Jaccard）≥ 0.3 → 保留
         （或数值冲突直接通过）
   层 6: 按 confidence × confidence × similarity 排序，每 section 留 top 6

4. 给每个 section 一次批量 LLM 调用（不是每个对调一次）：
   prompt: "下面 N 对 claim，对每对判断 COMPATIBLE/PARTIAL/CONTRADICTORY"

5. 系统按代码（不让 LLM 算）算 severity：
   severity_score = conf_a * conf_b * 数值差异度
   - score ≥ 0.7 → strong
   - 0.4-0.7 → moderate
   - < 0.4 → weak

6. 输出 sparse 字典 {section_id: [ConflictRecord, ...]}
   - 没分歧的 section 不出现（不是空列表）

7. writer 渲染时：
   - 如果 section 在字典里 → 在 prompt 里 conditional 注入冲突段
   - 不在 → prompt 完全不动（避免 anchor effect 让 LLM 硬编造分歧）
```

**④ 为什么这样**

- **只在 section 内检测，不跨 section**：跨 section 的"分歧"通常是设计意图（不同视角的拆分），不是问题。算了反而是噪声。
- **6 层过滤的目的是省 LLM 调用**：n 张卡片有 n*(n-1)/2 对，n=20 就是 190 对。让 LLM 看 190 对太贵。过滤后通常只剩 0-6 对。
- **严重度由代码算而非 LLM**：LLM 算 severity 容易飘（同样的卡片不同 prompt 给不同分数）。代码算确定性、可解释、可调参数。
- **写时条件注入而不是塞 "no conflicts: []"**：把空列表也写进 prompt，LLM 会出现 anchor 效应（"既然你说没冲突，那我也意识到要找冲突"），反而硬编造一些。直接不出现 + 也不在 prompt 里提"分歧"任何字眼最干净。

---

### 亮点⑥ 三层错误恢复（researcher 的鲁棒性核心）

**① 这是干嘛的**

让"某一个搜索失败"不会变成"整个研究任务失败"。

**② 不做会怎样**

任何一个 query 失败（network、rate limit、5xx）就抛异常 → 整个任务 fail → 用户重启 → 大概率同一个 query 又失败。

**③ 怎么做（流程级）**

```
当 researcher 处理某个 SubTask：

Tier 1 - 并发搜索 + 指数退避带抖动：
  - 用 rewritten_queries 并发搜（asyncio.gather）
  - 失败的 retriever 跳过（gather return_exceptions）
  - 至少一个有结果就成功
  - 全部 0 结果 → 进 Tier 2
  - 整体异常 → exponential backoff with full jitter, retry
    delay = uniform(0, base * 2^attempt)（带 jitter 避免雷鸣群效应）

Tier 2 - 多策略查询重写：（详见亮点②）
  - 1 次 LLM 调用产 3 个备选 query
  - 顺序试，第一个有结果即停
  - 全 0 → 进 Tier 3

Tier 3 - 优雅降级：
  - 不抛异常
  - SubTask status = "degraded"
  - degradation_reason = "No documents retrieved after retry and reformulation"
  - 继续处理同 section 的下一个 SubTask

后续：
  - reflector 会评估 degraded task 的影响（critical_gaps）
  - writer 会在报告里加 caveat（如果重要 section 大量降级）
```

**④ 为什么这样**

- **指数退避 + 抖动**：避免 N 个并发请求同时重试时再次撞同一个时间点（thundering herd）
- **Tier 不是平等**：Tier 1（重试）成本最低，Tier 2（LLM 重写）成本中，Tier 3（放弃）成本零。从便宜到贵依次试。
- **降级而不是失败**：deep research 任务包含 5-20 个 SubTask，1 个失败就整体 fail 太脆弱。**让能成功的都成功，能用的证据先用上**。

---

### 亮点⑦ Send API 并行 Section Research

**① 这是干嘛的**

让 N 个 section 的研究同时进行，不要串行。

**② 不做会怎样**

5 个 section 串行：每个 30 秒就是 150 秒。
5 个 section 并行：30 秒。

**③ 怎么做（流程级）**

```
LangGraph 的 Send API 用法：

1. 在 dispatch_sections 节点之后，有个 conditional edge:
   fan_out_sections()

2. fan_out_sections 的实现：
   - 按 section_id 把 pending_tasks 分组
   - 对每个 group 创建一个 Send("section_researcher", SectionResearchInput)
   - 返回 List[Send]

3. LangGraph 引擎收到 N 个 Send 后：
   - 同时启动 N 个 section_researcher 节点
   - 每个节点拿到自己的 SectionResearchInput（隔离的小 state）
   - asyncio 并发执行

4. 所有 N 个节点完成后：
   - LangGraph 自动用 reducer 合并它们的输出到主 state
   - knowledge_cards 用 operator.add（list append）
   - sub_tasks 用自定义 reducer _merge_sub_tasks（按 key 合并，
     状态优先级 completed > degraded > failed > pending）
   - 主 state 进入 collect_results 节点
```

**④ 为什么这样**

- **section 维度并行而不是 SubTask 维度并行**：同 section 的 SubTask 经常查相似的东西，并发会撞同一搜索引擎的限流。section 维度天然是不同主题，互不干扰。
- **隔离的 state 而非共享 state**：每个 Send 拿到自己的 input dict，互不污染。LangGraph 在节点结束时帮你合并。
- **reducer 决定合并策略**：默认是 list append（operator.add），自定义 reducer 可以做更精细的事（比如 sub_tasks 按 key 去重并保留最新状态）。

---

### 亮点⑧ Human-in-the-loop（draft + start 分离）

**① 这是干嘛的**

让用户在系统**真正花掉成本之前**有一次机会确认研究方向。

**② 不做会怎样**

用户问"调研 RISC-V"，系统理解成"调研 RISC-V CPU"（其实用户想要的是"调研 RISC-V 在数据中心场景"）。一口气跑完，几十次 LLM + 几百次搜索，最后产出的报告方向全错。用户傻眼，要重做。

**③ 怎么做（流程级）**

```
用户问题 → MCP 工具 draft_research_plan：
  1. 生成 seed queries
  2. 轻量侦察搜索（每个 query 3 条结果）
  3. LLM 草拟 execution_plan
  4. 渲染成 markdown，返回给用户
  5. **任务停在这里，不消耗任何后续成本**

用户看完决定：
  - 满意 → 调 start_research_task(task_id, "approve")
  - 想改 → 给 user_feedback 写改进意见，调 start_research_task
    (系统会基于 feedback 重新走 supervisor → 重新 draft)

  - 不满意又懒得改 → 干脆不调 start_research_task
    task 永远停在 draft 状态，没浪费任何资源

start_research_task：
  - 才真正调度 model worker 启动
  - 才开始执行真正烧 LLM/搜索的研究
```

**④ 为什么这样**

- **侦察 + draft 成本只有几次 LLM + 9 次搜索**（3 seed × 3 results）：用户即使不喜欢，也没烧掉多少额度
- **正式执行成本是几十次 LLM + 几百次搜索**：用户提前看一眼策略是不是合理，避免大规模返工
- **修改回路**：用户的 feedback 是结构化输入 → 系统能针对性调整。这比让用户重新提问要好

---

### 亮点⑨ 工程化加固（生产可用性）

这一组合不是某一个亮点而是一组配套：

| 加固 | 作用 |
|---|---|
| aiosqlite + WAL | 多任务并发读写无阻塞 |
| 每搜索引擎熔断器 | 单引擎抽风不影响其他引擎 |
| 每搜索引擎限流（asyncio.Semaphore） | 防止打满第三方限流 |
| Worker 子进程心跳 | 卡死自动检测 |
| Worker 重启滑动窗口预算 | 5min 内最多重启 5 次，防无限循环 |
| Worker job queue 有 maxsize | backpressure，防 OOM |
| 失败 task 持久化完整 traceback | 事后排查不用复现 |
| `deep-research-mcp prune` | 30 天前的终态任务清理 |
| `deep-research-mcp doctor` | 启动前配置自检 |
| `deep-research-mcp init` | 交互式 MCP 客户端配置向导 |

---

## 第 12 章 配置体系

### 12.1 配置读取的优先级

```
任意一个 setting 的值按以下优先级查找：

1. 环境变量（os.environ）→ 最高优先级
   例如 TAVILY_API_KEY=tvly-xxx
   这是 MCP 客户端通过 env 字段传进来的，标准做法

2. ~/.nanobot/config.json fallback
   - 读 providers 部分（如 deepseek 的 apiKey）
   - 读 tools.mcpServers[*].env 部分（如这个 server 的 env）
   - 这是为保留向后兼容（之前在 nanobot 里跑过的用户不用改）

3. 代码硬默认
   - 如果上面都没找到，用 dataclass 字段默认值
   - **API key 类的默认都是空字符串**——不允许硬编码 key 到代码里
```

### 12.2 启动自检

`deep-research-mcp doctor` 或 server 启动时都会自动跑 `Settings.validate_and_report()`：

```
[deep-research] [OK] LLM model=deepseek/deepseek-v4-flash (key via DEEP_RESEARCH_LLM_API_KEY)
[deep-research] [OK] Search engines active: tavily
[deep-research] [SKIP] exa: EXA_API_KEY not set
[deep-research] [WARN] DEEP_RESEARCH_LLM_API_KEY not set — relying on LiteLLM provider env vars
[deep-research] [ERROR] No search engine is available. Set TAVILY_API_KEY or enable DDG fallback.
```

### 12.3 关键 env 变量清单

按重要性排序：

**必须配的**：
- `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / 等：LLM 提供商 key
- `DEEP_RESEARCH_LLM_MODEL`：要用哪个模型（如 `deepseek/deepseek-v4-flash`）
- `TAVILY_API_KEY`：默认主力搜索（不配会回退到免费 DuckDuckGo）

**推荐配的**：
- `DEEP_RESEARCH_REPORT_DIR`：报告输出目录（默认 `~/Desktop/DeepResearch`）

**进阶（多源搜索）**：
- `EXA_API_KEY`、`SERPER_API_KEY`、`BOCHA_API_KEY`、`JINA_API_KEY`
- `DEEP_RESEARCH_MULTI_SOURCE=1` 启用多源聚合

**进阶（垂直学术）**：
- `DEEP_RESEARCH_ACADEMIC_SEARCH=1`
- `DEEP_RESEARCH_ARXIV_SEARCH=1`
- `DEEP_RESEARCH_PUBMED_SEARCH=1`

**进阶（可观测）**：
- `LANGSMITH_API_KEY`：开启 LangSmith 追踪

**进阶（行为微调）**：
- 4 项新功能开关（默认全开）：
  - `DEEP_RESEARCH_RECENCY_WEIGHTING=1`
  - `DEEP_RESEARCH_MAX_REFORMULATION_ATTEMPTS=3`
  - `DEEP_RESEARCH_CONFLICT_DETECTION=1`
  - `DEEP_RESEARCH_GROUNDED_CITATIONS=1`
- 各种半衰期、阈值、超时（具体见 settings.py）

---

## 第 13 章 接入与运维

### 13.1 安装

```bash
git clone https://github.com/studyzhige-ui/deep-research-mcp.git
cd deep-research-mcp
pip install -e .
```

可选额外：
- `pip install -e ".[local-models]"`：装本地 embedder + reranker
- `pip install -e ".[extras]"`：装 DuckDuckGo 兜底 + PyMuPDF（PDF 提取）

### 13.2 接入 MCP 客户端

**Claude Desktop**（`~/AppData/Roaming/Claude/claude_desktop_config.json`）：
```json
{
  "mcpServers": {
    "deep-research": {
      "command": "deep-research-mcp",
      "env": {
        "DEEPSEEK_API_KEY": "sk-...",
        "DEEP_RESEARCH_LLM_MODEL": "deepseek/deepseek-v4-flash",
        "TAVILY_API_KEY": "tvly-..."
      }
    }
  }
}
```

**Codex**（`~/.codex/config.toml`）：
```toml
[mcp_servers.deep-research]
command = "deep-research-mcp"
env = { DEEPSEEK_API_KEY = "sk-...", TAVILY_API_KEY = "tvly-..." }
```

**Claude Code**：用 `claude mcp add deep-research deep-research-mcp` CLI 添加。

重启客户端，工具就出现。

### 13.3 交互式向导（避免手写 JSON）

```bash
deep-research-mcp init
```

跟着问答走，最后输出可粘贴的客户端配置片段。**不写盘**，只显示在屏幕上。

### 13.4 维护 CLI

| 命令 | 作用 |
|---|---|
| `deep-research-mcp` | 启动 MCP server（stdio） |
| `deep-research-mcp doctor` | 不启动 worker 的配置自检 |
| `deep-research-mcp init` | 交互式生成客户端配置 |
| `deep-research-mcp prune` | 清理 30 天前的终态任务 |
| `deep-research-mcp prune --dry-run` | 预览要清理的内容 |
| `deep-research-mcp prune --days 7` | 改保留天数 |
| `deep-research-mcp --help` | 帮助 |

### 13.5 输出物

每个完成的任务在 `~/Desktop/DeepResearch/<task_id> - <topic>/` 下产生：

```
DeepResearch_Report_<task_id>.md      最终 Markdown 报告
DeepResearch_Cards_<task_id>.json     所有 KnowledgeCard
DeepResearch_Sources_<task_id>.json   引用列表（带编号）
DeepResearch_Activity_<task_id>.json  完整事件时间线
DeepResearch_Metadata_<task_id>.json  任务元数据（含 citation_audit）
DeepResearch_Process_<task_id>.log    详细日志
```

运行时数据在 `~/Desktop/DeepResearch/_runtime/`：

```
DeepResearch_TaskRegistry.sqlite         任务注册表
DeepResearch_GraphCheckpoints.sqlite     LangGraph state 快照
DeepResearch_WorkerBootstrap_*.log       worker 子进程启动日志
```

---

## 附录 A：核心数据结构速查表

### A.1 SubTask

| 字段 | 含义 |
|---|---|
| query | 原始搜索意图 |
| rewritten_queries | LLM 改写后的多个表达 |
| intent | 这个查询想填什么类型的信息 |
| section_id, section_title, section_goal | 属于哪个章节 |
| gap_type | definition / quant_support / counter_evidence / recent_update / detail / comparison |
| search_profile | max_results / search_depth / topic |
| status | pending / completed / failed / degraded |
| retry_count, last_error, degradation_reason | 错误恢复字段 |

### A.2 Document（规范化）

| 字段 | 含义 |
|---|---|
| document_id | URL 的 SHA1 hash |
| url, title, content, raw_content | 网页基本字段 |
| source_name | 来自哪个 retriever（tavily/exa/...） |
| source_layer | general / vertical |
| source_kind | web / paper / pdf / repo / news / mcp / custom |
| published_time, year, authors, venue, doi, pdf_url | 元数据 |
| score, search_quality_score | 排名分（含 recency 衰减） |
| recency_weight | 时效衰减权重，便于审计 |

### A.3 KnowledgeCard

| 字段 | 含义 |
|---|---|
| unit_id | 卡片 ID（U001, U002, ...） |
| section_id, evidence_id | 关联标识 |
| claim | 论点（一句话） |
| evidence_summary | 证据摘要 |
| exact_excerpt | 原文支持论点的具体句子（grounding 核心） |
| source, source_title, source_type | 来源 |
| confidence | high / medium / low |
| stance | supporting / counter / neutral / limitation |
| claim_type | fact / definition / procedure / trend / comparison / risk / metric |
| time_scope | historical / current / recent / future / timeless |
| entities | 命名实体列表 |
| evidence_strength | strong / medium / weak |
| evidence_score | 数值评分 |

### A.4 SectionDigest

| 字段 | 含义 |
|---|---|
| section_id, title, purpose, questions | 章节标识 |
| coverage_score, evidence_count_score, source_diversity_score | 质量分（0-1） |
| is_enough | 综合够不够 |
| review_reason | 不够的话原因 |
| missing_questions | 还没回答的子问题 |
| key_claims | 该章节最重要的几个论点 |
| items | 精选的 SectionDigestItem |

### A.5 ConflictRecord

| 字段 | 含义 |
|---|---|
| section_id, topic | 在哪一章、关于什么 |
| verdict | PARTIAL / CONTRADICTORY |
| claim_a, claim_b | 两个矛盾的论点 |
| source_a_url, source_b_url | 出处 |
| confidence_a, confidence_b | 各自置信度 |
| disagreement_summary | LLM 一句话总结分歧 |
| severity | strong / moderate / weak |
| severity_details | { score, confidence_product, divergence_factor, numeric } |

### A.6 citation_audit

| 字段 | 含义 |
|---|---|
| sections_total | 总章节数 |
| sections_grounded | 走 grounded JSON 路径的章节数 |
| sections_fallback | 降级到 free-form 路径的章节数 |
| citations_total | 全部引用数 |
| invalid_ids_dropped | LLM 试图编造 ID 被丢弃的次数 |
| ungrounded_paragraphs | 无引用段落数 |
| quote_failures | quote 字段验证失败次数 |
| numeric_failures | 数值校验失败次数 |
| per_section | 每章节详细 audit |

---

## 附录 B：性能/成本参考

典型一次研究（5 个 section，每个 3-5 个 SubTask）：

| 资源 | 用量 |
|---|---|
| 搜索 API 调用 | 50-150 次（通用 + 垂直） |
| Embedder 调用 | 1-2 次（每次几千句） |
| Reranker 调用 | 5-15 次（每次几百对） |
| LLM 调用 | 30-60 次（planner / researcher card extraction / reviewer / writer / conflict_detector） |
| LLM token 消耗 | 100k-500k tokens（视报告复杂度） |
| 总耗时 | 3-8 分钟（取决于 LLM 厂商响应速度） |

新增的 4 项功能对成本的额外开销：
- Recency 衰减：0（纯计算）
- 多策略查询重写：每次 0 结果触发时 +1 LLM 调用（典型 0-2 次）
- 跨源冲突检测：每个候选对足够的 section +1 LLM 调用（典型 2-4 次）
- Grounded citation：0 额外 LLM 调用（替换的不是新增）

合计每报告**多 2-6 次 LLM 调用**，但**全部由 max_reflection_loops 间接限制**——不会失控。

---

## 附录 C：技术栈速查

| 类别 | 用什么 |
|---|---|
| 工作流编排 | LangGraph |
| LLM 适配 | LiteLLM |
| MCP 协议 | FastMCP |
| 嵌入器 | bge-small-zh-v1.5（sentence-transformers） |
| 重排序器 | bge-reranker-base（CrossEncoder） |
| 向量检索 | FAISS（IndexFlatIP） |
| 文档解析 | readability-lxml + lxml + PyMuPDF |
| HTTP 客户端 | aiohttp + httpx |
| 异步 SQLite | aiosqlite |
| 进程通信 | multiprocessing.Queue + Manager.dict |
| 追踪 | LangSmith（可选） |
| 测试 | pytest + pytest-asyncio |

---

文档结束。

如需进一步细节，参考代码位置：
- 工作流定义：`deep_research_runtime/graph.py`
- 4 大 agent：`deep_research_runtime/agents/`
- 双层检索：`deep_research_runtime/search_service.py`
- Worker 子进程：`deep_research_runtime/worker.py`
- 4 项新功能：`recency.py` / `query_reform.py` / `conflict_detector.py` / `citation_grounding.py`
