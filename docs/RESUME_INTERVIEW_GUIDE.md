file_path: C:\Users\Alice\.nanobot\workspace\mcp_servers\docs\RESUME_INTERVIEW_GUIDE.md
content: # 简历面试逐句解析手册

> 目标：把简历上每一个 bullet、每一个数字、每一个技术名词都讲透，让你在面试里被任何细节追问都能流畅回答。
>
> 阅读方式：每一节都以"简历原文 → 用大白话先说一遍 → 关键术语解释 → 详细解释 → 高频追问 + 答案"的格式展开。
>
> 全文紧扣简历每一句话；最后一章自检清单会把简历从头到尾再过一遍确认全覆盖。

---

## 简历全文（作为索引基线）

> **项目介绍**：构建面向开放域复杂问题的自主深度研究系统，通过 MCP 协议暴露标准化工具接口，支持 AI 助手即插即用调用，完成从调研执行策略确认到结构化报告的完整闭环。
>
> 基于 LangGraph 构建以 Supervisor/Researcher/Reflector/Writer 为核心的 Workflow，采用 Map-Reduce 机制并行执行证据收集与结果汇总；Reflector 按研究任务评估证据覆盖度并生成补充检索任务，并结合跨轮次饱和度自适应路由至 Researcher 检索补强或 Writer 进行报告撰写。
>
> 设计双层检索架构（多源发现/垂直检索），在规划阶段引入多视角查询扩展，Researcher 层支持多策略查询重写（SIMPLIFY/SYNONYMS/DECOMPOSE/...，单次 LLM 调用产出 3 个备选），三层容错策略保障局部失败不影响整体质量；构建证据中间层，将检索结果抽取为带置信度标注的知识卡片，全局去重后按任务压缩为章节证据包，为大纲生成和分章节写作提供高质量上下文；对时效敏感任务按 publication date 做半衰期衰减加权，section 内检测跨源矛盾并要求 Writer 显式呈现分歧而非静默仲裁。
>
> 引入 Human-in-the-loop 审批机制，解耦调研执行策略与内容执行；Writer 节点基于大纲与证据包进行分章节报告撰写，采用封闭证据 ID 集 + verbatim quote / 数值容差校验的结构化 grounding 机制（对齐 Gemini Deep Research 设计），从生成端而非事后审计端消除引用幻觉，实现证据来源链接可追溯。
>
> 通过 LiteLLM 统一适配层接入多模型供应商；使用 SQLite 持久化图执行状态，支持长时任务恢复与增量追问；将向量嵌入与重排序剥离至独立 Worker 进程异步处理，避免主进程阻塞；任务存储采用 aiosqlite + WAL 支持并发读写；搜索层引入 per-engine 熔断 + 速率限制以隔离第三方故障。

---

# 第 1 章：项目介绍段

## 1.1 「构建面向开放域复杂问题的自主深度研究系统」

### 一句话先讲清

你做了一个**自动化研究助手**：用户丢一个开放式问题给它，它会自己拆任务、自己搜资料、自己读、自己判断够不够、自己写报告。

### 关键术语

- **开放域（open-domain）**：跟"封闭域"对比。封闭域比如"客服问答"——回答范围限定在某产品的 FAQ。开放域意思是用户可能问任何话题（RISC-V、量子计算、医美趋势...），系统不能预设范围。
- **复杂问题**：跟"单点查询"对比。单点查询是"巴黎首都"——一次搜索就答完。复杂问题是"调研 RISC-V 在数据中心 2025 的进展"——需要拆成几个方向、查多个来源、判断证据够不够、组织成报告。
- **自主（autonomous）**：跟"人工接力"对比。系统不是给一个"搜索结果列表"让人自己读，而是自己读完、自己写完报告。
- **深度研究（deep research）**：相对于"浅搜索"。浅搜索 = Google 给你一页链接。深度研究 = Perplexity / Gemini Deep Research / ChatGPT Deep Research 那种"自己规划 → 多轮搜索 → 写带引用报告"的范式。

### 详细解释

这句话定位的是项目的**整体能力等级**：不是"我做了一个 RAG"，不是"我做了一个搜索增强 chatbot"，而是"我做了一个跟 Perplexity 同类的产品"。

跟简单 RAG 的区别：
- 简单 RAG：1 次 query → 检索 top-k → 喂 LLM 写答案。一问一答。
- 深度研究：多轮规划 + 多次检索 + 多轮反思 + 大纲驱动写作 + 引用校验。一问一份报告。

### 高频追问

**Q1：你说"自主深度研究系统"，但市面上 Perplexity 和 ChatGPT Deep Research 都已经是 SaaS 产品了，你做这个的意义是什么？**

A：定位不一样。Perplexity 是托管在他们服务器上的 SaaS，你用他们的 LLM、他们的搜索引擎、他们的数据。我做的是**自部署的开源 MCP 服务器**——可以挂到用户自己的 AI 助手里（Claude Desktop、Cursor、Codex），用户用自己的 API key、查自己想查的源、跑在自己电脑上、数据不出本地。这是隐私敏感场景 / 企业内网场景 / 想自己定制研究流程的场景的痛点。另外作为开源项目，工作流可见、可改、可调试，对学习和二次开发友好。

**Q2："开放域"具体怎么体现的？系统怎么处理任何话题？**

A：体现在三个设计上：
1. **不预设知识库**：所有证据都是实时搜索来的，不依赖向量数据库。
2. **双层检索动态路由**：planner 根据 query 推断该用通用搜索（Tavily/Exa）还是垂直搜索（Semantic Scholar/arXiv/PubMed）。比如查"AI 论文综述"自动启用学术垂直层。
3. **多策略查询重写**：通用 query 失败时，researcher 用 5 种策略（SIMPLIFY/SYNONYMS/DECOMPOSE/BROADEN_TIME/SPECIFY）改写。能适应用户写法千差万别。

**Q3："复杂问题"和"简单问题"的边界在哪？什么样的问题不适合这个系统？**

A：界限是"需不需要多轮迭代"。
- 适合：调研类（"X 领域 2025 进展"）、对比类（"A 和 B 谁更适合 Z 场景"）、综述类（"X 技术的优缺点"）。
- 不适合：精确查询（"巴黎人口"，浪费）、计算类（"2+2"）、强时效（"现在股价"，搜索延迟太大）。

设计上有 `time_scope` 字段判断，如果是 timeless 的事实查询，系统会用更轻量的路径（少轮反思、少备选 query）。

---

## 1.2 「通过 MCP 协议暴露标准化工具接口」

### 一句话先讲清

系统的对外接口走 MCP 标准协议，不是自定义 HTTP API。

### 关键术语

- **MCP（Model Context Protocol）**：Anthropic 主导的开源协议，让 AI 助手可以调用外部工具。类比 USB 标准——任何符合 MCP 的工具都能"插"到任何符合 MCP 的 AI 助手上。
- **stdio transport**：MCP 的一种通信方式，工具进程通过标准输入输出跟 AI 助手通信。优点：不暴露网络端口，安全；缺点：本地进程，不能跨机器。
- **标准化工具接口**：每个工具有固定的 JSON Schema（参数、返回类型），AI 助手能自己读懂"这个工具是干嘛的、需要什么参数"。

### 详细解释

具体来说，本项目对外暴露 **7 个 MCP 工具**：

| 工具 | 干嘛的 |
|---|---|
| `check_research_runtime` | 检查运行时是否就绪（worker 在跑吗？key 配了吗？） |
| `draft_research_plan` | 输入研究问题 → 输出策略草稿 + task_id |
| `start_research_task` | 用户确认后启动正式研究 |
| `get_research_status` | 查进度 |
| `get_research_result` | 拿结果 |
| `follow_up_research` | 追问 |
| `compare_report_versions` | 比较两个版本 |

实现细节：用 Python 的 FastMCP 库声明这些工具（`@mcp.tool()` 装饰器），FastMCP 帮我把每个函数包装成符合 MCP 协议的 JSON-RPC endpoint。

### 高频追问

**Q1：MCP 跟你自己写一个 REST API 比有什么优势？为什么不直接用 FastAPI？**

A：3 点优势：
1. **AI 助手原生集成**：Claude Desktop、Cursor、Codex 等都内置 MCP 客户端，用户加配置就能用，不需要写 glue code。换成 FastAPI 用户得自己写"AI 助手调用 HTTP API"的对接逻辑。
2. **工具发现自动化**：MCP 协议要求工具声明 schema，AI 助手能自动读懂工具用法。FastAPI 的 OpenAPI 也有 schema，但需要 AI 助手专门去读，没标准化。
3. **stdio 安全**：MCP 默认 stdio 通信，不开网络端口，更安全。FastAPI 必须监听端口，要考虑认证、TLS。

**Q2：你说"标准化"，但每个 MCP 客户端的配置格式都不一样（Claude Desktop 是 JSON、Codex 是 TOML），这算什么标准化？**

A：配置格式确实因客户端而异（这是客户端层的事），但 **MCP 协议层完全统一**——工具声明、调用、返回的 JSON-RPC 消息格式各家一致。我项目里只实现一份服务端代码，能被 Claude Desktop、Cursor、Codex 等所有 MCP 客户端调用，不需要为每个客户端重写。这就是协议层标准化的价值。配置格式差异我用了 `deep-research-mcp init` 交互式向导解决，问几个问题输出多种客户端能粘贴的片段。

**Q3：MCP 用 JSON-RPC 而非 gRPC 或 REST，为什么？**

A：是 Anthropic 设计协议时的选择，不是我选的。但理由可以理解：JSON-RPC over stdio 简单到极致——不需要 TLS、不需要端口、不需要负载均衡，AI 助手 fork 一个子进程就能直接通信。gRPC 要 protobuf、要端口、要 HTTP/2，stdio 跑不起来。REST 也要端口。MCP 想要"零配置即插即用"，stdio + JSON-RPC 是最简单的实现。

---

## 1.3 「支持 AI 助手即插即用调用」

### 一句话先讲清

用户不用写代码，只需把这个工具的命令名写到 AI 助手配置里，重启就能用。

### 关键术语

- **即插即用（plug and play）**：跟 USB 设备类比。USB 鼠标插上去，操作系统自动识别、自动加载驱动、立刻能用。这个项目也是：用户在 AI 助手配置文件里加一行 `"command": "deep-research-mcp"`，重启就能用了。

### 详细解释

具体怎么"即插即用"：

1. **统一命令行入口**：pyproject.toml 里 `[project.scripts]` 注册了 `deep-research-mcp` 命令。用户 `pip install` 后 PATH 里就有这个命令。
2. **stdio 启动**：AI 助手只需 spawn 一个子进程跑 `deep-research-mcp`，通过 stdin/stdout 通信。**不开端口、不要 systemd、不要 Docker**。
3. **配置即环境变量**：所有 API key（LLM 厂商、搜索 API）都通过 MCP 客户端配置的 `env` 字段传入。没有"安装后还要手动改配置文件"。
4. **配置自检**：`deep-research-mcp doctor` 命令一键查看哪些 key 配了、哪些没配、哪些引擎能用。

用户接入完整流程（5 分钟）：

```
当用户想用这个工具，他首先会：
1. pip install -e . （从 GitHub clone 后安装）
2. 跑 deep-research-mcp init，按提示输入 LLM key 和搜索 key
3. 向导输出一段可粘贴的 JSON / TOML
4. 用户把这段粘贴到自己 AI 助手的 mcpServers 配置里
5. 重启 AI 助手
6. 在对话里输入"用 deep-research 调研 X"，AI 助手自动调工具
```

### 高频追问

**Q1："即插即用"听起来很好，但用户还是要装 Python、装依赖、配 key，这算即插即用吗？**

A：相对于"自己写脚本调用 API"已经是即插即用了。诚实地说：
- 真正的"即插即用"是 SaaS（注册账号点几下就用上）
- 我这个是开源自部署的，必须有 Python 环境是前提
- 但我做了 3 件事让它尽可能简单：
  - 单条 `pip install -e .` 完成安装
  - `deep-research-mcp init` 交互式向导，不用手写 JSON
  - `deep-research-mcp doctor` 一键自检，配错了立刻看出来
- 跟同类开源项目（GPT Researcher、autogpt 这种）比，已经是接入最简单的之一了

**Q2：如果用户的 AI 助手不支持 MCP（比如自研的）怎么办？**

A：两条路：
1. **自己写 MCP 客户端**：MCP 协议规范公开，几百行 Python 就能实现一个基本客户端。
2. **绕过 MCP 直接用 Python 类**：项目内部是清晰的分层架构，`DeepResearchService` 类暴露了所有能力。可以直接 `from deep_research_runtime.service import DeepResearchService; svc = DeepResearchService(); await svc.tool_draft_plan(...)`。MCP 只是套了一层壳。

**Q3：即插即用的代价是什么？有什么是 SaaS 能做、你这个做不到的？**

A：3 件事：
1. **跨设备同步**：SaaS 的研究历史能在手机/电脑同步。我的数据存本地 SQLite，跨设备要自己想办法。
2. **大规模并发**：SaaS 后端有 N 台机器扛并发。我的是单机进程。
3. **统一更新**：SaaS 改完所有人立刻用上。我的得用户自己 `git pull`。

但对应也有好处：数据不出本地，API key 不被第三方拿走，可以定制 / fork。

---

## 1.4 「完成从调研执行策略确认到结构化报告的完整闭环」

### 一句话先讲清

整条链路是一个闭环：用户提问 → 系统出策略 → 用户确认 → 系统执行 → 系统出报告。不是一步走完。

### 关键术语

- **闭环（closed loop）**：跟"单步操作"对比。单步是"用户 query 进、报告出"。闭环是"中间会停下、跟用户确认、用户可修改、再继续"。
- **调研执行策略**：系统在真正搜索之前先想好"我打算怎么查"——查哪些方向、用什么搜索引擎、什么样的源算可信。这是一份结构化文档，叫 `ResearchExecutionPlan`。
- **结构化报告**：不是一段流水文字，是带章节、引用、来源列表的 Markdown 文档。
- **完整闭环**：覆盖从提问到拿到最终报告中间所有步骤，没有"中间需要用户跳到别的工具完成某一步"这种缺口。

### 详细解释

闭环的 7 个阶段：

```
1. 用户提问                    ← MCP 工具 draft_research_plan
2. 系统侦察 + 草拟策略         ← 内部 planner.draft_execution_plan
3. 系统把策略给用户            ← 返回 markdown
4. 用户确认 / 修改             ← Human-in-the-loop 暂停点
5. 系统执行正式研究            ← MCP 工具 start_research_task
6. 用户查进度 / 拿结果         ← MCP 工具 get_research_status / get_research_result
7. 用户追问                    ← MCP 工具 follow_up_research（回到步骤 5 的增量版）
```

"完整"指 7 个阶段都在系统内部解决，用户不需要中途切到别的工具。

### 高频追问

**Q1：为什么必须搞"策略确认"这一步？让系统直接跑完不就行了？**

A：3 个理由：
1. **成本控制**：正式研究要 30-60 次 LLM 调用 + 100+ 次搜索。如果系统误解了用户问题（比如把"RISC-V 数据中心"理解成"RISC-V CPU 综述"），跑完才发现方向不对，等于白烧一份。策略确认这步只用几次 LLM 调用 + 9 次轻量搜索（3 seed query × 3 results），成本极低，能在大成本之前及时纠偏。
2. **用户控制感**：研究类任务用户对方向有偏好。比如同样"调研 RISC-V"，做硬件设计的关注芯片架构，做云原生的关注数据中心部署。让用户选方向比让系统瞎猜好。
3. **可解释性**：用户看到策略才知道系统打算怎么干。如果直接给报告，质量差时不知道是策略不对还是执行不到位。

**Q2："结构化报告"具体长什么样？跟普通的 LLM 摘要有什么区别？**

A：结构化体现在 4 点：
1. **章节明确**：报告由 5-8 个 `## 章节标题` 组成，每个章节有明确主题。章节顺序由 outline_builder 根据真实证据决定。
2. **引用可追溯**：每个论断后面有小角标 `<sup>[3]</sup>`，点击跳转到原始 URL。报告末尾有完整 References 列表。
3. **多个产物**：除了 Markdown 报告，还输出 `cards.json`（所有知识卡片）、`sources.json`（来源目录）、`activity.json`（执行时间线）、`metadata.json`（含 citation_audit）。
4. **可对比**：每次追问产生新 version，可以用 `compare_report_versions` 看版本差异。

普通 LLM 摘要：一段连续文字，没引用、没结构、没追溯。

**Q3：闭环里如果某一步失败（比如搜索全挂、LLM 超时）系统怎么处理？**

A：分层降级：
- **单 query 失败**：3 层错误恢复（详见 3.4），最终降级为 status=degraded，不影响其他 query
- **某 section 大量降级**：reflector 标记 critical_gaps，writer 在该 section 加上"证据不充分"caveat
- **某次 LLM 调用失败**：retry + 退避（指数 + jitter）
- **整个图执行超时**：拿现有的卡片走 writer，能写多少写多少
- **整个任务崩溃**：state 在 SQLite checkpoint，重启后从最后 checkpoint 恢复，**不丢之前的卡片**

---

# 第 2 章：LangGraph 核心 Workflow 段

## 2.1 「基于 LangGraph 构建以 Supervisor/Researcher/Reflector/Writer 为核心的 Workflow」

### 一句话先讲清

整个研究流程是一张用 LangGraph 画的"流程图"，4 个核心节点对应"领头人 / 干活的 / 质检员 / 写报告的"。

### 关键术语

- **LangGraph**：一个 Python 库，让你用"图"的方式定义 AI 工作流。每个节点是一个函数，节点之间有边表示数据流转。优势：支持循环、并行、checkpoint、条件路由。
- **Workflow / 工作流**：一个明确的"先做什么再做什么"的执行流程。
- **Supervisor**：领头人节点，负责把抽象的研究策略拆成可以并行执行的具体任务清单。
- **Researcher**：实际"动手干活"的节点，每个 Researcher 实例负责一个 section 的搜索 + 抽取卡片。
- **Reflector**：反思节点，看一眼当前收集到的证据够不够，决定是否继续搜或者收尾。
- **Writer**：最终写报告的节点，把所有证据组织成 Markdown 文档。

### 详细解释

完整的 LangGraph 图实际上有 8 个节点，但**主干语义就是 4 个核心节点**，其他都是辅助管道：

```
核心 4 节点              辅助节点
                                                   
Supervisor    ──→        dispatch_sections  ← 把 supervisor 输出 fan-out 给 N 个 researcher
                                ↓
                         (Send×N) 并行
                                ↓
Researcher ×N ──→        collect_results    ← reduce N 个并行结果到主 state
                                ↓
Reflector     ──→        条件路由
              ↑↓                ↓
              loop          outline_builder  ← 由证据驱动生成大纲
                                ↓
                         detect_conflicts   ← 跨源冲突检测
                                ↓
Writer        ──→        END
```

每个核心节点的职责：

| 节点 | 输入 | 做什么 | 输出 |
|---|---|---|---|
| Supervisor | execution_plan | 把策略里的 query_strategy 展开成 SubTask 列表，给每个 SubTask 加工 rewritten_queries / search_profile | sub_tasks |
| Researcher | 一组 SubTask | 搜索 + 内容提取 + 嵌入排序 + LLM 抽卡片 + 去重 | knowledge_cards |
| Reflector | 所有 knowledge_cards | 评估每 section 覆盖度，算饱和度，决定下一步 | quality_review + 可能的补研究任务 |
| Writer | section_digests + plan_data | 按章节调用 LLM 写 markdown，渲染引用 | final_report |

### 高频追问

**Q1：为什么用 LangGraph 而不是简单写个 `async def workflow(): ...` 的函数？**

A：4 个理由：
1. **状态持久化**：LangGraph 的 AsyncSqliteSaver 自动给每个节点执行完写一次 state 快照。任务跑到一半进程崩了，重启后从最后一个 checkpoint 继续。手写 async 函数要自己实现这套。
2. **条件路由 + 循环**：Reflector → 回 Researcher 这条循环边，手写 async 会变成嵌套 while + 状态管理，复杂且容易出 bug。LangGraph 用 `add_conditional_edges` 一行声明。
3. **并行节点 + 自动合并**：Send API 让多个 researcher 并行运行，自动用 reducer（operator.add 或自定义函数）合并结果到主 state。手写要管 asyncio.gather + 手动合并 list。
4. **可观察**：LangGraph 集成 LangSmith，每个节点的输入输出自动 trace，方便事后排查。

**Q2：4 个核心节点的边界是怎么划的？为什么不是 3 个或 5 个？**

A：按"研究的不同认知阶段"划：
- **拆任务**：Supervisor。研究开始前的"思考怎么干"
- **干活**：Researcher。中间最重的"搜索 + 读 + 抽取"
- **质检**：Reflector。中间的"够不够 + 该继续还是收尾"
- **写报告**：Writer。最后的"组织成文"

为什么不合并：
- 合并 Supervisor + Researcher：那"拆任务"和"干活"在同一节点，没法干净支持循环（reflector 决定补研究时，得回到拆任务还是回到干活？）
- 合并 Reflector + Writer：那"判断够不够"和"写"绑死，没法支持"够了但 outline 还没生成"的场景

为什么不分得更细（比如 Reviewer 和 Reflector 分开）：可以分，但语义重叠多（都是"评估"），划得越细越难记。4 个是平衡可读性和职责单一的折中。

**Q3：核心 4 节点之外还有 dispatch / collect / outline_builder / detect_conflicts 这些辅助节点，简历为什么不提？**

A：因为简历空间有限，**主干（核心 4 节点）才是高阶设计**，辅助节点是把主干串起来的"管道工"。但面试如果被问到细节，可以补充：
- dispatch_sections + collect_results：Send API 并行的两端
- outline_builder：证据驱动的大纲生成（commit 中标记为 Improvement，独立节点是为了让大纲在证据**之后**生成）
- detect_conflicts：commit 3 新增节点，专门负责跨源冲突检测

这些节点合并到核心节点也能跑，但**单独成节点的好处是各自可独立测试、可独立 trace、可独立开关**。

---

## 2.2 「采用 Map-Reduce 机制并行执行证据收集与结果汇总」

### 一句话先讲清

N 个 section 同时让 N 个 researcher 并行干活（Map），然后等所有 researcher 都做完再汇总到一起（Reduce）。

### 关键术语

- **Map-Reduce**：分布式计算的一种范式。Map 阶段：把大任务拆成 N 个小任务并行做；Reduce 阶段：把 N 个小结果合并成大结果。Hadoop / Spark 的核心思想。
- **Send API**：LangGraph 提供的并行机制。一个节点可以输出多个 `Send(target_node, payload)`，LangGraph 引擎会同时启动 N 个 target_node 实例。
- **State 合并 / Reducer**：N 个并行节点同时修改主 state 时，要告诉 LangGraph 怎么合并。比如 `knowledge_cards` 用 `operator.add`（列表拼接），`sub_tasks` 用自定义 reducer（按 key 去重保留最新状态）。

### 详细解释

整个 Map-Reduce 流程：

```
当 supervisor 把 sub_tasks 准备好后：

【Map 阶段】
1. dispatch_sections 节点执行
2. fan_out_sections 这个条件边把 sub_tasks 按 section_id 分组
3. 每组对应一个 Send("section_researcher", {section_id, pending_tasks, ...})
4. LangGraph 引擎收到 N 个 Send（N = section 数），同时启动 N 个 section_researcher 实例
5. 每个 researcher 拿到自己的隔离 input（不是共享主 state，避免污染）

【N 个 Researcher 并行执行】
6. 各自独立做：搜索、内容补全、worker 嵌入、LLM 抽卡片、去重
7. 各自返回 {knowledge_cards: [...], sub_tasks: [...], section_results: [...]}

【Reduce 阶段】
8. LangGraph 引擎根据 ResearchState 的 reducer 注解合并：
   - knowledge_cards: operator.add → 所有 researcher 的卡片列表 append 到一起
   - sub_tasks: _merge_sub_tasks（自定义 reducer）→ 按 query 去重，状态优先级 completed > degraded > failed > pending
   - section_results: operator.add
9. 合并后的主 state 进入 collect_results 节点
10. collect_results 做最后一次全局去重（跨 section 的同 URL+claim 卡片）
```

### 高频追问

**Q1：你说 Map-Reduce 并行，但 N 个 researcher 同时调搜索 API 会不会撞同一个限流？**

A：会，所以我加了两层保护：
1. **每搜索引擎独立 Semaphore 限流**：`search_engine_rate_limit=4`，每个引擎同时最多 4 个并发请求。N 个 researcher 加起来超过 4 也只能排队。
2. **每搜索引擎独立熔断器**：连续失败 3 次后 cooldown 60s，cooldown 期间所有 researcher 都跳过它。

效果：并行的是 researcher 节点，但**对外部 API 的实际并发受限流和熔断器控制**，不会真的把 API 打爆。

**Q2：N 个 researcher 修改同一份 knowledge_cards 列表，会不会冲突？**

A：不会，因为每个 researcher 拿到的是**自己的隔离 state**（通过 Send 传入的 SectionResearchInput）。它的返回值是 `{knowledge_cards: [本 section 的卡片]}`，不是修改主 state。然后 LangGraph 引擎用 reducer（这里是 `operator.add`）把所有 researcher 的返回值合并到主 state。所以**没有共享可变状态，没有竞争条件**。

**Q3：如果其中一个 researcher 跑得特别慢（比如卡在某个慢搜索 API），整个 Map 阶段是不是都得等它？**

A：是的，这是 Map-Reduce 模式天然的"长尾问题"。我做了几个缓解：
1. **per-engine 限流 + 熔断**：避免单个 researcher 反复重试卡死。
2. **指数退避有 cap=30s**：单次 retry 最长 30s，不会无限拖。
3. **3 层错误恢复**：还是不行就 status=degraded，不阻塞主流程。
4. **integer task timeout**：整个 graph 执行有总超时（`task_execution_timeout`，默认 5400s），到点强制结束，拿到啥写啥。

理论上还可以做"动态 timeout"——快的 researcher 完成后给剩下的设个 hard deadline。当前没做，因为 deep research 任务允许较长延迟，这个优化收益不大。

---

## 2.3 「Reflector 按研究任务评估证据覆盖度并生成补充检索任务」

### 一句话先讲清

Reflector 不是简单"是否够"的二元判断，而是按**每个 section 单独评估**——这个 section 覆盖度多少？缺什么问题？这些缺口怎么补？

### 关键术语

- **证据覆盖度（coverage）**：某 section 提出的子问题里，**有多少被收集到的卡片回答了**的比例。0-1 之间。比如 section 有 5 个子问题，3 个被卡片回答了，coverage = 0.6。
- **补充检索任务（follow-up task）**：reflector 发现 section 缺什么后，**自动构造新的 SubTask** 让下一轮 dispatch 派给 researcher。比如发现"X 模型的 benchmark 数据"没回答，就生成新 SubTask 专门查 benchmark。

### 详细解释

reflector 的评估是**两层结合**——规则评审 + LLM 评审 + LLM 否决权。

**规则评审（rule_based_section_review）**：

| 维度 | 计算公式 | 满分条件 |
|---|---|---|
| coverage_score | 被回答的子问题数 / 总子问题数 | 1.0 |
| evidence_count_score | min(1.0, 卡片数 / 4) | 卡片 ≥ 4 张 |
| source_diversity_score | min(1.0, 不同域名数 / 3) | 域名 ≥ 3 个 |
| primary_source_score | primary_source 卡片数 / 总数 | 大部分是一手来源 |
| claim_type_diversity_score | min(1.0, 不同 claim_type / 4) | 论点类型多样 |

**is_enough 判断**：
```
覆盖 ≥ 0.7 AND 卡片数 ≥ 3 AND 域名数 ≥ 2 AND 主要来源占比 ≥ 0.25
```

**LLM 评审（llm_section_review）**：

把 section 摘要 + top 卡片给 LLM，让它判断：
- semantic_coverage_score：语义上是否足够回答 section 核心问题
- support_score：证据支撑论点的强度
- conflict_score：证据之间是否有冲突
- missing_questions：列出还没回答的子问题
- weak_claims：标出薄弱论点
- follow_up_requests：具体应该补什么

**合并 + 否决权**：如果 LLM 说"语义不够"，强制把规则评审的 is_enough 改为 False。**LLM 有否决权但没否决权之外的权力**——它只能否决，不能批准（规则说不够它也不能改）。

**生成补充任务**：对每个 is_enough=False 的 section：
- 把 missing_questions、gap_types、required_source_types 打包成新的 SubTask
- 让 planner 给这些 task 补充 rewritten_queries、search_profile
- append 到 sub_tasks，下一轮 dispatch 派出去

### 高频追问

**Q1：为什么规则评审 + LLM 评审两层结合？只用一层不行吗？**

A：单层都不够鲁棒。
- **只用规则**：会被"卡片数量很多但都是低质量重复"骗过——5 张说同样事的卡片，coverage 可能很高，但实质上证据是单点的。
- **只用 LLM**：LLM 评审有时候过于宽容（"看起来似乎够了"），有时候过于挑剔（"我觉得还缺这个那个"），方差大。
- **两层结合**：规则评审定一个保守底线（卡片数、域名数等硬指标必须达到），LLM 评审有否决权（语义不够直接停）——**降低单点失效风险**。

**Q2：补充检索任务怎么生成？是直接复制原 SubTask 吗？**

A：不是简单复制。对每个 missing_question，系统会：
1. 用规则推断 gap_type（这个缺口是 definition / quant_support / counter_evidence / recent_update / detail 哪种）
2. 根据 gap_type 选 evidence_goal（要补什么类型的证据）
3. 由 planner.prepare_subtasks_for_search 加工：生成 rewritten_queries、search_profile
4. 标注 required_source_types / required_evidence_types

所以是**有针对性的补充查询**，不是无脑重搜。

**Q3：如果 reflector 反复说"还不够"，会不会无限循环？**

A：有 3 层 safety net：
1. **max_reflection_loops**（默认 3）：硬上限，最多反思 3 轮，到了强制走 outline_builder
2. **min_loops_before_early_stop**（默认 1）：至少 1 轮才能停（避免极简单问题第 0 轮就停）
3. **饱和度阈值**：详见下一节，当本轮提升 + 新增都小到一定程度，强制停

实际跑下来 90% 的任务在 1-2 轮就停了。

---

## 2.4 「并结合跨轮次饱和度自适应路由至 Researcher 检索补强或 Writer 进行报告撰写」

### 一句话先讲清

Reflector 不只看"是否还有缺口"，还看"本轮研究比上一轮有没有显著进展"。如果连续两轮没进展（饱和），停下。

### 关键术语

- **跨轮次（across iterations）**：多次反思之间。第 1 轮反思完决定继续 → 跑第 2 轮 researcher → 第 2 轮反思看"跟第 1 轮比进展多少"。
- **饱和度（saturation score）**：0-1 的数字。1 = 完全饱和（没新东西）；0 = 严重不足。
- **自适应路由**：不是固定"跑满 3 轮"，而是**动态决定**——简单问题 1 轮就停，难问题最多跑 3 轮。

### 详细解释

饱和度公式：

```
coverage_delta = 当前 coverage - 上一轮 coverage   ← 本轮覆盖度提升了多少
marginal_gain = 本轮新增卡片数 / 总卡片数         ← 本轮新增证据占比

saturation = 1.0 - (coverage_delta * 0.6 + marginal_gain * 0.4)
```

**意义**：如果 coverage_delta 和 marginal_gain 都接近 0，说明这轮研究没带来啥实质提升 → saturation 接近 1 → 该停了。

**路由决策**（在 `should_stop_early` 函数里）：

```
当 reflector 评估完所有 section：

1. 算 saturation_score 和 has_follow_ups
2. 综合判断：
   - 如果 loop_count < min_loops (默认 1) → 必须继续，路由 Researcher
   - 如果 has_follow_ups == False → 没缺口了，路由 Writer
   - 如果 saturation_score >= threshold (默认 0.85) → 饱和了，路由 Writer
   - 如果 loop_count >= max_loops (默认 3) → 硬上限，路由 Writer
   - 其他情况 → 继续，路由 Researcher
```

**关键设计**：
- **权重 60/40**：coverage_delta 比 marginal_gain 重要，因为"新增 30 张卡但没回答新问题"是没意义的扩张。
- **小阈值 0.85**：deep research 任务允许少量未覆盖（永远完美不现实）。

### 高频追问

**Q1：饱和度阈值 0.85 是怎么定的？测过吗？**

A：诚实回答：这是**经验值，不是用数据集精调的**。理由：
- 0.95 太严：意味着 coverage_delta 和 marginal_gain 加起来 < 0.067，几乎要求"零增长"才停。实际跑下来很多任务永远停不下来，被 max_loops 兜底。
- 0.7 太松：第 2 轮一点点进展就被判为饱和，质量不稳定。
- 0.85 折中：实测大部分任务 1-2 轮就停，少数复杂任务跑满 3 轮才停。

如果要严肃做这个数：需要构造 50+ 个 golden set 问题，跑不同阈值，人工标注每份报告"够好"，看 F1 最高的阈值。我没做这个，是项目的 known limitation。面试时如果被追问，可以坦诚说"阈值是设计参数，没做大规模评测"。

**Q2：权重 60/40 怎么来的？为什么不是 50/50？**

A：跟阈值一样是经验值。但 60/40 的逻辑解释：
- **coverage 是"回答问题"的指标**：直接衡量"用户问题答得多全"
- **marginal_gain 是"卡片增长"的指标**：可能是有用的新论点，也可能是冗余的重复
- 所以 coverage 应该权重更高。但 marginal_gain 也不能完全忽略——有时候新卡片虽然不直接回答新问题，但**加深了论证强度**（比如多了个权威来源支持已有论点），值得继续。
- 60/40 的意思：每提升 1 个百分点 coverage = 每提升 1.5 个百分点 marginal_gain。

**Q3：自适应早停有没有可能"该停时没停"或"不该停时停了"？怎么验证？**

A：两种错误模式都存在：
- **该停时没停**（false positive on "需要继续"）：浪费 LLM 调用，但报告质量不会差。代价是钱和时间。
- **不该停时停了**（false negative on "需要继续"）：报告残缺，coverage 没拉到位。代价是质量。

我做了两层防护：
1. **min_loops_before_early_stop=1**：至少跑 1 轮 reflector 才能停，不会第 0 轮就停。
2. **LLM 否决权**：即使规则说够了，LLM 评审说不够也强制继续。

但**没有严格验证**——没用 golden set 测量过 P(stop|enough) 和 P(continue|not_enough)。这也是 known limitation。

---

# 第 3 章：检索 + 证据中间层 + 时效 + 冲突段

## 3.1 「设计双层检索架构（多源发现/垂直检索）」

### 一句话先讲清

搜索分两层：第 1 层是通用网页搜索（Tavily/Exa/Serper...），第 2 层是垂直专业搜索（Semantic Scholar/arXiv/PubMed），根据 query 自动决定要不要启用第 2 层。

### 关键术语

- **双层检索**：层级化的搜索体系，跟"只用一个搜索引擎"对比。
- **多源发现（general layer）**：用多个通用搜索引擎广撒网，找各种来源的网页。
- **垂直检索（vertical layer）**：用专业领域的搜索 API 找特定类型的资料（学术论文、医学文献、专利等）。
- **Retriever**：搜索器的统称。每个搜索引擎一个 Retriever 类。

### 详细解释

**通用层**（9 个 retriever）：

| Retriever | API | 特点 |
|---|---|---|
| Tavily | api.tavily.com | AI 优化，返回正文，**默认主力** |
| Exa | api.exa.ai | 神经网络语义搜索 |
| Serper | serper.dev | Google 包装 |
| Bocha | bochaai.com | 中文语义 |
| SerpAPI | serpapi.com | 备选 Google |
| Bing | api.bing.microsoft.com | 微软 |
| Google | googleapis.com | Custom Search |
| Searx | 自建 | 元搜索 |
| DuckDuckGo | ddgs 库 | **免费兜底** |

**垂直层**（3 个 retriever）：

| Retriever | 用途 |
|---|---|
| SemanticScholar | 学术论文 + 引用数 |
| ArXiv | 物理/CS 预印本 |
| PubMedCentral | 生物医学论文 |

**双层路由**：

```
当 researcher 收到一个 SubTask：

1. SearchService.infer_verticals(query) 检查 query 文本：
   - 含 "paper" / "journal" / "arxiv" / "论文" / "期刊" / "学术" 等 → 触发 academic 垂直层
   - 不含上述关键词 → 仅用通用层

2. 通用层并行调用所有配了 key 的 retriever
3. 如果触发了垂直层，并行调用 SemanticScholar + arXiv + PubMed
4. 所有 retriever 结果合并，按 search_quality_score 排序去重
```

### 高频追问

**Q1：为什么要做"双层"？一层（只用通用搜索）不够吗？**

A：通用搜索对学术内容**召回率低**：
- 用 Tavily 搜"transformer 论文综述"，第一页大概率是博客科普文，不是真正的论文
- 而 Semantic Scholar 直接搜会返回 arXiv 论文、引用数、作者
- 两个搜索的**质量分布完全不一样**

所以根据 query 类型动态启用垂直层，相当于**针对学术 query 走专门通道**，回报远好于通用层。

**Q2：12 个 retriever 是不是太多？维护成本怎么办？**

A：诚实回答：**多但有用**：
- 用户只会激活其中几个（默认只用 Tavily + DuckDuckGo）
- 但提供 12 种是为了让不同环境的用户能用上自己已经有 key 的：装了 Bing/Serper 套餐的就用那个，企业 Searx 内网就用 Searx，等等
- 每个 retriever 实现是独立的小 class（几十行），共享的 Retriever Protocol，**维护负担分散**
- 都有熔断器和限流器，挂掉一个不影响其他

如果做精：可以删到 3-4 个核心（Tavily + Exa + SemanticScholar + DDG）。但作为开源项目，宽兼容性是吸引用户的关键。

**Q3：垂直检索是怎么用关键词触发的？如果 query 没明显关键词但其实是学术问题怎么办？**

A：当前用的是**关键词匹配**（infer_verticals 里硬编码了一组词），简单粗暴但有效。

边缘情况处理：
- planner 在 draft 阶段会让 LLM 标注 `source_strategy`（要不要走学术）。即使 query 文本没关键词，LLM 推断的策略也会触发垂直层。
- 也可以走 SubTask 的 `source_types` 字段——researcher 拿到 SubTask 时显式知道这是学术任务。

**没做的**：用嵌入/分类器判断 query 类型。这是一个可改进点——但当前关键词路由对 80% 场景够用，没紧迫到要上模型分类。

---

## 3.2 「在规划阶段引入多视角查询扩展」

### 一句话先讲清

planner 在草拟策略时，不只让 LLM 写一组 query，而是让 LLM 从多个角度想 query。

### 关键术语

- **多视角（multi-perspective）**：跟"单一角度"对比。同一个问题"AI 安全"，从"研究者视角"会问"对齐技术进展"，从"政策视角"会问"监管现状"，从"工程师视角"会问"漏洞案例"。
- **查询扩展（query expansion）**：把一个用户原始 query 扩展成多个具体可执行的搜索 query。

### 详细解释

planner 多视角扩展有两个阶段：

**阶段 1：构造 research tracks（研究方向）**：
- 给 LLM 用户问题 + reconnaissance 结果
- 让 LLM 输出 4-6 个研究方向（query_strategy），每个方向是不同视角

比如"调研 RISC-V 数据中心进展"会得到：
- 方向 1：芯片厂商进展（视角：硬件厂家）
- 方向 2：数据中心实际部署案例（视角：终端用户）
- 方向 3：性能 benchmark 对比 ARM/x86（视角：性能评测）
- 方向 4：生态系统支持（视角：软件支持）
- 方向 5：商业可行性 + 投资动态（视角：商业）

**阶段 2：每个方向的 query 矩阵**：
- supervisor 把每个方向展开成 2-3 个 SubTask
- 每个 SubTask 由 planner.prepare_subtasks_for_search 进一步加工出 2-4 个 rewritten_queries

最终 1 个用户问题 → 4-6 个方向 → 10-20 个 SubTask → 30-50 个具体搜索 query。

### 高频追问

**Q1：怎么保证 LLM 给出的多视角真的"多"而不是 4 个相似方向？**

A：3 个手段：
1. **prompt 显式要求**：planner prompt 里写"please ensure each research track approaches the topic from a clearly different angle"
2. **execution_plan schema 里有 source_strategy 字段**：每个方向标注用通用 / 垂直 / 混合，强制类型差异
3. **rule-based filter**：planner 输出后系统检查方向数 ≥ 3、每个方向至少 1 个 SubTask

**没做的**：用嵌入相似度过滤太接近的方向。这是可优化点。

**Q2："多视角"跟后面的"多策略查询重写"是一回事吗？**

A：**不是**，两个层级：
- **多视角扩展（planner 阶段）**：解决"研究方向覆盖广"。是策略层的事，决定"查哪几个方向"。
- **多策略查询重写（researcher 阶段）**：解决"单 query 召回不到"。是执行层的事，当某个具体 query 失败时换 3 种说法重试。

简历这里讲的是**前者**——在规划阶段就保证多角度。

**Q3：多视角扩展有没有可能导致方向太散，最终报告主题模糊？**

A：有这个风险。3 个机制控制：
1. **execution_plan 由用户确认**：Human-in-the-loop，用户看到 5 个方向觉得太散可以让系统调整
2. **outline_builder 后置**：研究做完后再生成大纲，**outline 不一定按 5 个 research tracks 来**——LLM 看完所有证据后，可能合并某两个方向、删除某个证据不足的方向。
3. **reviewer 标 critical_gaps**：如果某个方向证据严重不足，reviewer 标出来，writer 加 caveat 而不是强行写。

---

## 3.3 「Researcher 层支持多策略查询重写（SIMPLIFY/SYNONYMS/DECOMPOSE/...，单次 LLM 调用产出 3 个备选）」

### 一句话先讲清

researcher 搜到 0 结果时，1 次 LLM 调用让模型一口气给出 3 种不同思路改写的 query，顺序试。

### 关键术语

- **查询重写（query reformulation）**：原 query 失败时，换一种写法再试。
- **SIMPLIFY**：策略 1，去掉具体定语，更泛化。"2025 年 RISC-V 在数据中心的落地" → "RISC-V data center"
- **SYNONYMS**：策略 2，换同义术语、中英互译。"大语言模型" → "LLM" / "foundation model"
- **DECOMPOSE**：策略 3，复合问题拆开。"A 和 B 谁更适合 C" → "A on C" + "B on C"
- **单次 LLM 调用产出 3 个**：1 次 prompt 让 LLM 同时写 3 个备选 query，不是 3 次 prompt 各写 1 个。

### 详细解释

5 种策略全集（默认启用前 3 种）：

| 策略 | 解决什么失败 | 例子 |
|---|---|---|
| SIMPLIFY | query 太长太具体 | "2025 年 RISC-V 服务器芯片性能 benchmark" → "RISC-V server benchmark" |
| SYNONYMS | 关键术语不匹配索引词 | "大语言模型" → "LLM" |
| DECOMPOSE | 复合问题搜不到完整答案 | "A 和 B 对比" → "A info" + "B info" |
| BROADEN_TIME | 时间限定过严 | "2025 RISC-V" → "RISC-V"（去时间词） |
| SPECIFY | query 太模糊 | "AI" → "GPT-4" |

**完整流程**：

```
当 researcher 发现某个 query 搜到 0 结果：

1. 单次 LLM 调用，prompt 明确枚举 3 种策略，要求每种各写一个备选：
   "原查询: {query} 返回 0 结果。生成 3 个备选：
    1. SIMPLIFY: 去除细节，更通用
    2. SYNONYMS: 换关键术语
    3. DECOMPOSE: 拆成 2 个子问题
    只返回 JSON: {queries: [...]}"

2. LLM 返回 {"queries": ["alt1", "alt2", "alt3"]}

3. researcher 顺序试这 3 个 query：
   - 第一个有结果 → 用它，记录 result["reformulated_query"] = "alt1"，停
   - 全部 0 结果 → 进入 Tier 3 降级

4. LLM 调用次数：固定 1 次（不管尝试几个备选）
```

### 高频追问

**Q1：为什么单 LLM 调用而不是 3 次？**

A：成本和延迟：
- 3 次串行 LLM 调用 = 3 倍延迟 + 3 倍 token 成本
- 1 次 LLM 调用让模型一口气写 3 个备选 = 跟原来 1 次成本一样，只是 output 多一点 token

**关键洞察**：策略多样性靠 prompt 强制（"每种策略各写一个"），不靠多次调用。

**Q2：3 个备选数字是怎么定的？为什么不是 5 个？**

A：**3 是默认值，可通过 `DEEP_RESEARCH_MAX_REFORMULATION_ATTEMPTS` 配置到最多 5**。3 的理由：
- 前 3 个策略（SIMPLIFY/SYNONYMS/DECOMPOSE）覆盖最常见的失败模式
- 第 4 个 BROADEN_TIME 仅对时效任务有用
- 第 5 个 SPECIFY 罕见（query 太模糊的情况少）
- 更多 = LLM 输出更长 + 试错次数更多 = 边际收益递减

**测量方法 / 基线对比**（诚实说）：
- 测量：没用 golden set 测过"3 vs 5 的召回率"差异
- 基线：跟旧版本（1 个备选，单一 SIMPLIFY 策略）对比，主观跑过 10+ 个测试 query，**3 备选时召回明显提升**——尤其中英混合 query（SYNONYMS 策略救场）
- 想严肃量化需要构造 zero-result query 数据集，跑 N=1/3/5 备选对比成功率

**Q3：3 个备选顺序试，如果第 1 个就成功，后 2 个白生成不浪费吗？**

A：会浪费一点输出 token，但**生成成本对比"再调 1 次 LLM 的成本"小一个数量级**：
- 单次 LLM 调用：input ~500 tokens + output ~50 tokens = ~$0.0001
- 多生成 2 个备选：output 多 ~30 tokens = ~$0.00001
- 而省去的"重新调用 LLM"是 ~$0.0001 量级

所以**生成 3 个并联试，远比"先调 LLM 拿 1 个，没用又调 LLM 拿 1 个"省**。

---

## 3.4 「三层容错策略保障局部失败不影响整体质量」

### 一句话先讲清

researcher 一个 query 失败有 3 种"恢复路径"，从轻到重：自动重试 → 换 query → 标记降级。

### 关键术语

- **容错策略（fault tolerance）**：系统遇到错误时不崩溃，按预设方案继续。
- **局部失败**：某个 SubTask 失败（比如 query 搜不到、API 5xx），跟"整体失败"（整个任务崩了）对比。
- **三层**：Tier 1 retry / Tier 2 reformulation / Tier 3 degrade。

### 详细解释

完整 3 层逻辑：

**Tier 1 — 带指数退避 + 抖动的重试**：
```
当 researcher 用 rewritten_queries 发起搜索：

1. 并发调多个搜索引擎（asyncio.gather）
2. 失败的引擎返回 [] (gather return_exceptions)
3. 至少一个有结果 → 成功，进入卡片抽取
4. 全部 0 → 进 Tier 2

如果整体异常（网络全断等）：
- 指数退避 delay = base * 2^attempt
- + 抖动 jitter = uniform(0, delay)   ← 避免雷鸣群（多 query 同时重试撞同一时刻）
- 重试最多 N 次（默认 2）
```

**Tier 2 — 多策略查询重写**：
- 详见 3.3
- 1 次 LLM 调用 → 3 个备选 query → 顺序试
- 第一个有结果即停
- 全 0 → 进 Tier 3

**Tier 3 — 优雅降级**：
- 不抛异常、不阻塞主流程
- SubTask 标记 status="degraded"
- 写 degradation_reason="No documents retrieved after retry and reformulation"
- 跳到同 section 下一个 SubTask 继续

**后续处理**：
- reflector 评估 degraded task 的影响（assess_degraded_impact）
- 如果某 critical section 大量降级，标记 critical_gaps
- writer 在该 section 加 caveat（"本节证据不充分，存在以下缺口"）

### 高频追问

**Q1：为什么必须是 3 层不是 2 层或 4 层？**

A：**3 层正好对应 3 类不同失败模式**：
- **Tier 1 解决 transient failure**（瞬时失败）：网络抖动、API 5xx、限流。重试就行。
- **Tier 2 解决 vocabulary failure**（词汇失配）：query 表达不对、术语不通用。改 query 解决。
- **Tier 3 解决 capacity failure**（能力上限）：这个话题真的没资料，或者所有可用源都挂了。承认失败但不阻塞。

如果只有 2 层（retry + 降级），漏掉"词汇失配"——很多 query 重试 10 次也搜不到，但换个说法立刻有结果。

如果有 4 层（再加一层"用 LLM 直接生成事实"），那就变成了"幻觉"，违背 grounded research 原则。

**Q2：指数退避带 jitter 解决什么问题？没 jitter 行不行？**

A：解决"雷鸣群效应"（thundering herd）：
- 5 个并行 SubTask 同时遇到 API 503
- 都进入 Tier 1 重试
- 没 jitter：5 个 task 第 1 次重试都在 t+2s 撞上，第 2 次都在 t+4s 撞上 → 同样的并发再次撞墙
- 有 jitter：5 个 task 在 [0, 2s] 内随机时刻重试 → 错开打散，对面 API 缓过来

具体公式：`delay = uniform(0, base * 2^attempt)`（"full jitter"，AWS 推荐）。

**Q3：Tier 3 降级会让报告残缺吗？怎么让用户知道哪些是降级的？**

A：会让某些 section 证据少，但**不至于报告残缺**，因为：
1. **reflector 评估降级影响**：assess_degraded_impact 函数判断哪些 critical section 受影响
2. **writer 加 caveat**：在受影响 section 开头明确写"本节涉及 X 子问题缺少充分证据"
3. **citation_audit 暴露统计**：get_research_result 输出里有 sections_with_degraded_tasks 字段，用户看一眼就知道

更严重的话题（critical section 50%+ 任务降级），writer prompt 会被指示"对该 section 整体降低 confident 表述、增加 limitations 段落"。

---

## 3.5 「构建证据中间层，将检索结果抽取为带置信度标注的知识卡片」

### 一句话先讲清

搜来的整篇网页太碎太杂，系统把它转成结构化的"知识卡片"——一张卡片 = 一个论点 + 出处 + 置信度。

### 关键术语

- **证据中间层（evidence intermediate layer）**：在"原始网页"和"最终报告"之间加的一层数据结构。
- **知识卡片（KnowledgeCard）**：结构化的最小证据单元。
- **置信度（confidence）**：high / medium / low 三档，由 LLM 抽卡片时打。基于"证据强度 + 来源类型 + 数据具体性"综合判断。

### 详细解释

KnowledgeCard 完整 schema：

| 字段 | 含义 |
|---|---|
| unit_id | 唯一 ID（U001, U002, ...） |
| section_id | 关联到哪个研究 section |
| claim | 论点（一句话） |
| evidence_summary | 证据摘要 |
| **exact_excerpt** | 原文中支持该论点的具体句子（grounding 核心） |
| source | 来源 URL |
| source_title | 来源标题 |
| source_type | primary_source / secondary_source / analysis / community |
| **confidence** | high / medium / low |
| stance | supporting / counter / neutral / limitation |
| claim_type | fact / definition / procedure / trend / comparison / risk / metric |
| time_scope | historical / current / recent / future / timeless |
| entities | 命名实体列表 |
| evidence_strength | strong / medium / weak |
| evidence_score | 数值评分 |

**抽取流程**：

```
当 researcher 拿到 worker 返回的 evidence（句子级证据）：

1. 构造 LLM prompt：
   - 给 LLM evidence_records 列表（带 evidence_id）
   - prompt 要求：从这些 evidence 中抽取卡片，每张卡片必须用 evidence_ids 指明来源
   - 要求输出 confidence、stance、claim_type 等结构化字段

2. LLM 返回 JSON: {cards: [{claim, evidence_summary, evidence_ids, confidence, ...}]}

3. researcher 后处理（normalize_quality_cards）：
   - 校验每张卡片的 evidence_ids 必须在 allowed evidence 集合里
   - 分配 unit_id（U001, U002...）
   - 填补缺失字段
   - 跟当前 KnowledgeCache 比对去重
```

### 高频追问

**Q1：为什么要加"证据中间层"？直接把网页喂给 writer 不行吗？**

A：3 个理由：
1. **上下文爆炸**：单个 section 可能搜到 20 篇网页，每篇 5000 字 = 100K 字符。Writer prompt 直接放不下。
2. **缺乏结构**：网页里有大量不相关内容（导航、广告、评论）。LLM 写报告时容易被无关内容带偏。
3. **没有 grounding 锚点**：直接喂网页时，writer 引用某个论断没法精确指出"这句话出自这篇文章第几段"。卡片的 exact_excerpt 就是精确锚点。

**Q2：置信度（confidence）是怎么打的？LLM 凭什么判断？**

A：LLM 抽卡片的 prompt 里有具体规则：
- **high**：一手来源（官方文档、原论文）+ 具体数据 + 明确表述
- **medium**：二手来源（科技媒体）+ 大致信息
- **low**：博客观点 + 主观判断 + 间接转述

LLM 看 evidence 的来源类型 + 表述具体性综合判断。

**诚实说**：LLM 打分有方差，不同 LLM、不同 prompt 会打不同分。但有 source_type 字段做兜底——source_type=primary_source 的卡片自动倾向于 high confidence。

**Q3：抽取卡片时 LLM 会不会"美化"原文？比如把"可能"改成"确定"？**

A：这就是 grounding 要解决的问题。3 层防护：
1. **prompt 显式要求**：抽卡时要求 LLM 保留原文措辞强度（probabilistic vs certain）
2. **exact_excerpt 必须是原文**：LLM 必须从 evidence 里抠出原句子，不是用自己的话总结
3. **commit 4 的 grounded generation（详见 4.4）**：writer 写报告时强制引用 exact_excerpt，进一步约束

但有个 known gap：抽卡时 LLM 的 paraphrase 可能漂移。如果非常严格可以加一层"卡片 claim vs exact_excerpt 的 entailment 校验"，目前没做。

---

## 3.6 「全局去重后按任务压缩为章节证据包」

### 一句话先讲清

跨 section 之间也可能有重复卡片（同一篇文章被两个 section 用到），最后再做一次全局去重；然后把每个 section 的卡片压缩成一个 SectionDigest（章节证据包）。

### 关键术语

- **全局去重**：跨 section 的去重，跟"section 内去重"对比。
- **章节证据包（SectionDigest）**：单个 section 的卡片集合的压缩版，只保留 key_claims、coverage_score、是否够用、top 几张卡片。
- **按任务压缩**：每个 section 有自己的研究任务，压缩时根据该 section 的 questions 和 evidence_requirements 选择最相关的卡片。

### 详细解释

**两层去重**：

```
section 内去重（KnowledgeCache，研究阶段）：
  - 每个 section_researcher 实例独立的 cache
  - URL + claim 的 MD5 hash 一致 → 丢弃
  - 保证同 section 不重复处理

全局去重（collect_results 节点）：
  - 5 个 section 各自返回卡片列表，合并到主 state
  - 跨 section 再过一遍 MD5 hash 去重
  - 保证最终卡片集合无重复
```

**章节证据包构造**（reviewer.build_section_digest）：

```
当 reflector 评审完某个 section：

1. 拿到该 section 的所有卡片（可能 10-20 张）
2. 选 top N（默认 24）按 evidence_score 排序
3. 提取 key_claims（top 6 个论点，去重）
4. 计算 coverage_score、evidence_count_score 等
5. 标 missing_questions、follow_up_requests
6. 构造 SectionDigest 对象

效果：原始 20 张卡片（约 20000 字符）→ digest（约 3000 字符）
```

### 高频追问

**Q1：为什么要两层去重而不是一层？**

A：性能 + 局部性：
- **section 内去重在搜索过程中实时跑**：避免同 section 反复搜同一篇文章浪费 API
- **全局去重在收尾时一次性跑**：跨 section 的重复不影响搜索过程，集中处理一次成本低
- 两层结合：**section 内即时反馈**（避免内部浪费）+ **全局兜底**（最终交付干净）

**Q2：SectionDigest 压缩会丢信息吗？writer 看不到完整卡片怎么办？**

A：writer 看到的是**两个东西**：
1. **SectionDigest**（压缩摘要）：用来快速理解 section 整体情况
2. **raw_cards**（精选完整卡片）：每个 section 选 top 6-8 张完整卡片传给 writer，writer 用这些做精确引用

所以**信息没完全丢**，digest 是"目录"，raw_cards 是"详情"。Writer 两个都看。

**Q3：去重是简单 MD5 hash 还是语义去重？语义重复（同样意思不同表述）怎么办？**

A：诚实说：**当前是 MD5 hash 去重，不是语义去重**。
- 优点：快、可解释、零误删
- 缺点：同义论点不会被去重（A 网页说"X 准确率 95%"和 B 网页说"X 在测试中达到 95%"会保留两张卡片）

但**这是 feature 不是 bug**：
- 同义论点的多源出现 = corroborating evidence（互相佐证），应该保留，提升论点 confidence
- 如果两个源说同一论点是矛盾的（一个 95% 一个 80%），冲突检测会捕捉到（详见 3.9）

真要做语义去重需要嵌入 + 聚类，引入新风险（误删 corroborating evidence），权衡下不值得。

---

## 3.7 「为大纲生成和分章节写作提供高质量上下文」

### 一句话先讲清

证据中间层（卡片 + digest）的最终目的就是给 outline_builder 和 writer 提供 LLM-friendly 的输入。

### 关键术语

- **高质量上下文**：相对于"原始网页堆砌"，结构化、精炼、有锚点的上下文。
- **大纲生成（outline generation）**：决定报告章节怎么排（先讲背景 → 技术现状 → 应用案例 → 展望）。
- **分章节写作（per-section writing）**：每个章节独立调一次 LLM 写，最后拼接。

### 详细解释

为什么"高质量"：

| 维度 | 低质量上下文（直接喂网页） | 高质量上下文（卡片 + digest） |
|---|---|---|
| 长度 | 100K+ tokens | 3-5K tokens |
| 信号噪声比 | 低（含导航、广告） | 高（只留 claim + excerpt） |
| 结构化 | 自由文本 | 字段化（claim, source, confidence, stance...） |
| 引用锚点 | 没 | 每张卡片有 source + exact_excerpt |
| LLM 决策成本 | 高（要自己判断哪些重要） | 低（已按 evidence_score 排过） |

**对 outline_builder 的好处**：基于 section_digests 决定大纲 → 大纲一定能配上证据（不会"想讲 X 但没资料"）。

**对 writer 的好处**：每个 section 只看自己的 digest + raw_cards → 上下文聚焦、不容易跑偏、引用精确。

### 高频追问

**Q1：3-5K tokens 数字怎么来的？跟 100K 怎么比的？**

A：实际跑下来的经验值，不是精算：
- **digest 长度**：每个 section 包含 key_claims（最多 6 条） + items（最多 24 条卡片摘要），每条 ~120 字符 = ~3000-4000 字符 / section
- **5 个 section 总和**：15-20K 字符 ≈ 3-5K tokens

对比直接喂网页：
- 5 个 section 各 4 篇网页，每篇 5K 字 = 100K 字符 ≈ 25-30K tokens
- 直接超过大多数模型的有效上下文窗口（即使 200K 窗口的模型，深度处理 100K 输入也会"注意力稀释"）

**Q2：上下文压缩会不会让 LLM"想引用某个细节但找不到"？**

A：会有这风险，但有兜底：
- digest 里有 items 字段（精选卡片，含 exact_excerpt）
- writer 写 section 时拿到的不只是 digest，还有 raw_cards（更完整的 top 6-8 张卡片）
- 真要某个细节查不到，应该是它本来就没被 reflector 选进 digest（说明 evidence_score 低）

如果用户事后觉得"少了某个重要细节"，可以走 follow_up_research 增量补充。

**Q3：高质量上下文需要 LLM 多次调用产出（抽卡片本身就要 LLM），跟"直接喂网页"比成本怎么样？**

A：**总成本反而更低**：
- 直接喂：1 次 100K input 的 LLM 调用，input cost 高
- 中间层：N 次抽卡片调用（每次小 input，5K tokens 左右）+ 1 次写报告调用（5K input）

LLM 定价里 **input token 比 output 便宜很多**，但**100K 输入比 N × 5K 累积更贵**。再加上注意力质量下降问题，**中间层方案 ROI 明显更好**。

---

## 3.8 「对时效敏感任务按 publication date 做半衰期衰减加权」

### 一句话先讲清

时效任务（"2025 最新"类）下，旧文档的打分自动降低——一年前的文档权重只有现在的一半，两年前只有四分之一。

### 关键术语

- **时效敏感任务**：planner 会给任务标 time_scope。值有 recent / current / current_year / future / timeless / historical。前 4 个属时效敏感。
- **publication date**：文档的发表时间。从 search API 返回的 published_time / year 字段提取。
- **半衰期衰减（half-life decay）**：跟放射性衰变同款数学模型。每过一个半衰期，权重减半。

### 详细解释

公式：

```
weight = 0.5 ^ (age_months / half_life_months)
最终 score = base_score × weight
```

**half_life_months 表**（按 time_scope）：

| time_scope | half-life |
|---|---|
| recent | 6 个月 |
| current / current_year | 18 个月 |
| future | 12 个月 |
| 其他（默认） | 36 个月 |
| timeless / historical | ∞（不衰减，weight=1.0） |

**关键边界处理**：
- 未知日期 → weight = 1.0（不惩罚 metadata 缺失，那是另外的问题）
- 未来日期（爬虫脏数据）→ clamp 到 age=0，weight=1.0
- 极旧文档 → floor 在 0.3（不能完全抹掉一份 2010 年综述）
- 支持多种日期格式：ISO 8601 / "2024" / "2024-03" / "Mar 15, 2024" / 整数年

### 高频追问

**Q1：半衰期 6/18/36 月这些数字怎么定的？**

A：经验值，**基于"信息陈旧速度"的直觉**：
- 新闻类（recent）：半年前的就过时，6 个月一半权重合理
- 大趋势类（current_year）：去年的还有参考价值，18 个月折半合理
- 综合类（默认 36）：3 年前的文档仍有效，但 6 年前的折半

**测量方法 / 基线**：诚实说，**没做严肃 A/B 评测**。这是设计参数。如果要严格调，需要：
1. 构造时效敏感的 golden set（"2025 最新 X"类问题）
2. 加上 / 不加 recency 衰减跑同一任务
3. 人工评估报告"信息时效性"评分
4. 调半衰期对比

当前的诚实定位：**有总比没有好**。旧版本完全没考虑时效，新版本至少消除了"2019 综述打败 2025 预印本"这类明显问题。

**Q2：为什么 floor 在 0.3 而不是 0？**

A：极旧文档仍可能有价值——比如 2010 年的深度学习综述对理解"为什么深度学习兴起"有参考。floor 0.3 保证这种文档不被完全抹掉，仍能进入候选集，但不会盖过 2024 预印本。

floor 设 0 的风险：完全抹掉一些"经典老文献"，让报告失去历史视角。

设 0.5 的风险：太宽松，时效衰减效果减弱。

0.3 是经验折中。

**Q3：publication date 提不到怎么办？很多网页没有元数据。**

A：fallback 链：
1. 各搜索 API 返回的 published_time / published_date 字段
2. SemanticScholar 等垂直 API 返回的 year 字段
3. 都没有 → weight = 1.0（不惩罚）

**关键设计**：缺失日期不能当作"超旧"处理。那样会变成"元数据完整度的惩罚"，对很多 small blog 不公平。**只惩罚明确老的，不惩罚不明确的**。

---

## 3.9 「section 内检测跨源矛盾并要求 Writer 显式呈现分歧而非静默仲裁」

### 一句话先讲清

同一个 section 里两个源说的不一样（A 说 95%，B 说 80%），系统**自动检测出来**，让 writer 在报告里**显式写**"Sources disagree on X"，而不是 writer 自己挑一个写。

### 关键术语

- **跨源矛盾（cross-source conflict）**：两张卡片论点不一致。
- **显式呈现分歧**：报告里明确说"两个来源不一致"，而不是隐藏分歧。
- **静默仲裁**：writer 看到两个矛盾的源时，自己"觉得"该信哪个，写出来的内容是一边倒的。这是项目要避免的反模式。

### 详细解释

**完整流程**（详见 conflict_detector.py）：

```
当 detect_conflicts 节点执行：

1. 按 section 分组卡片，跳过 section 内卡片少于 min_cards（默认 3）的

2. 对每 section 内卡片做 6 层过滤生成候选对：
   层 1: 全部两两组合
   层 2: 同 URL 的对去掉（一篇文章不会自相矛盾）
   层 3: time_scope 不重叠的去掉（2020 数据 vs 2024 数据不是矛盾）
   层 4: stance 检查：
         - supporting × counter → 保留
         - 或 同实体 + 数值差异 > 2% → 保留
         - 其他 stance 组合 → 去掉
   层 5: claim 文本相似度 ≥ 0.3（Jaccard）→ 保留
   层 6: 按 confidence_product 排序，每 section 留 top 6

3. 每 section 一次批量 LLM 调用：
   "对下面 N 对 claim，分类：COMPATIBLE / PARTIAL / CONTRADICTORY"

4. 严重度由代码算（不让 LLM 算）：
   severity = conf_a × conf_b × 数值差异度
   - score ≥ 0.7 → strong
   - 0.4-0.7 → moderate
   - < 0.4 → weak

5. 输出 sparse 字典 {section_id: [ConflictRecord, ...]}
   - 没分歧的 section 不出现（不是空列表）

6. writer 写该 section 时：
   - 如果 section 在字典里 → prompt 里注入 conflict block
     "下面的 claim 对来自该 section，存在分歧，请显式写'sources disagree on X'"
   - 不在字典里 → prompt 完全不动（不出现 conflict 字眼）
```

### 高频追问

**Q1：为什么 6 层过滤这么复杂？1 层不行吗？**

A：**控制 LLM 调用成本**。
- n 张卡片有 n(n-1)/2 对，n=20 就是 190 对
- 让 LLM 看 190 对太贵且效果差（LLM 注意力分散）
- 6 层过滤通常把 190 对缩到 0-6 对
- 然后 1 次 LLM call 看这 0-6 对，便宜准确

每一层过滤都有具体目的：
- 层 2（同 URL）：明显假阳性（一篇文章不会自相矛盾）
- 层 3（time_scope）：时间序列 vs 矛盾
- 层 4（stance）：neutral × supporting 没语义冲突
- 层 5（相似度）：不同主题的卡片不算"矛盾"
- 层 6（cap 6）：硬上限防爆炸

**Q2：严重度为什么由代码算不让 LLM 算？**

A：3 个理由：
1. **稳定性**：LLM 算 severity 容易飘——同样的卡片不同 prompt 给不同分数。代码算确定性。
2. **可解释**：代码公式可以拿出来给用户看"为什么这个标 strong"。LLM 黑盒。
3. **省 LLM 调用**：LLM 只做"判断是否冲突"，不做"评级"，prompt 更聚焦，质量更高。

代码算法：`score = conf_a × conf_b × 数值差异度`。
- conf_a/b 用 {"high": 1.0, "medium": 0.6, "low": 0.3} 映射
- 数值差异度：> 20% → 1.0；5-20% → 0.6；< 5% → 0.3；非数值 → 0.5

**Q3：writer 怎么"显式呈现"分歧？会不会让报告读起来很乱？**

A：通过 prompt 模板控制：
- 有分歧 section：prompt 里加 conflict block，要求 writer 用 "Sources disagree on X: [a] reports Y, while [b] reports Z" 句式
- writer 输出的报告里会自然出现这种段落
- 严重度 weak → 一句带过；strong → 单独一小段展开

**关键设计**：**有分歧时才提，没分歧时 prompt 完全不动**。否则会出现 anchor 效应——LLM 看到 "no conflicts: []" 的占位也会"想找冲突"，可能编造。

---

# 第 4 章：HITL + Grounded Citation 段

## 4.1 「引入 Human-in-the-loop 审批机制」

### 一句话先讲清

系统不会"一上来就跑完"，中间有一个"等用户确认策略"的暂停点。

### 关键术语

- **Human-in-the-loop（HITL）**：流程中间有需要人工确认的步骤。
- **审批（approval）**：用户对系统草拟的策略点头或修改。

### 详细解释

HITL 的具体形态：

```
当用户在 AI 助手里问 "调研 RISC-V 数据中心"，系统：

第 1 阶段（自动，~30 秒）：
1. tool_draft_plan 被调用
2. planner 生成 seed queries
3. 轻量侦察搜索（3 query × 3 results）
4. LLM 草拟 execution_plan
5. 返回 markdown 策略 + task_id 给用户

【暂停点】系统不再做任何事，等用户输入

第 2 阶段（用户操作）：
6. 用户在 AI 助手里看到策略
7. 满意 → 让 AI 助手调 tool_execute_plan(task_id, "approve")
8. 想改 → 在 user_feedback 里写改进意见
9. 不喜欢 → 干脆不调 → 任务永远停在 draft 状态

第 3 阶段（用户确认后自动，~3-8 分钟）：
10. tool_execute_plan 触发后台 LangGraph 流程
11. 执行 supervisor → researcher → reflector → writer
12. 完成后存报告，等用户来取
```

**关键点**：第 1 阶段只用 1-3 次 LLM 调用 + 9 次搜索（成本极低），第 3 阶段才是大头消耗。HITL 让用户在大消耗之前及时纠偏。

### 高频追问

**Q1：HITL 增加了用户操作步骤，体验不会变差吗？**

A：trade-off。**研究类任务里，用户对"方向是否准确"特别敏感**，跑错一份报告要重做花的时间远多于看一眼策略花的时间。所以多一步审批，长远是节省用户时间。

如果用户嫌烦，可以教 AI 助手"先 draft 再立刻 approve"（不真的看策略），相当于退化成无 HITL 模式。但默认设计是有 HITL 的。

**Q2：用户给 feedback 怎么处理？系统会真的根据 feedback 改吗？**

A：feedback 是结构化输入到 supervisor。流程：
1. tool_execute_plan 收到 user_feedback="改成只查 2024 年以后的，去掉 GitHub 项目类源"
2. 把 feedback 传给 supervisor 节点
3. supervisor 重新调 planner.draft_execution_plan，prompt 里附加 feedback
4. 新的 execution_plan 会反映 feedback 要求

**实现 detail**：当前是"重新生成 plan"，没做 incremental edit。理论上 LLM 看到原 plan + feedback 能更精准修订，但 incremental 容易 prompt 漏改，整体重做更稳。

**Q3：HITL 跟"agent 自主性"是不是矛盾？真正的 AI agent 不是应该全自动？**

A：哲学上不矛盾，是层级问题：
- **任务策略层 HITL**：用户对方向有偏好，应该确认 → 这是好的
- **任务执行层全自动**：策略确认后，搜索、抽卡、写报告完全自主 → 这是好的

完全自动反而是反模式——deep research 不是"接 chatbot 闲聊"，方向错了代价大。HITL 是该有的 guardrail。

可以类比"自动驾驶"：策略层（去哪儿）人决定，执行层（怎么开）AI 决定。

---

## 4.2 「解耦调研执行策略与内容执行」

### 一句话先讲清

"用什么策略查"和"按这个策略真的去查"是两个分开的阶段，由两个分开的 MCP 工具触发。

### 关键术语

- **解耦（decoupling）**：两件事分开，互不影响。
- **调研执行策略**：用什么搜索引擎、查哪些方向、怎么筛选证据。是"方法"。
- **内容执行**：按这个方法真的搜索、读、写。是"动作"。

### 详细解释

解耦体现在 3 层：

**第 1 层：工具级解耦**
- `tool_draft_plan` 只产策略，不执行
- `tool_execute_plan` 只执行，不再思考策略
- 中间是 SQLite 持久化的 task 状态

**第 2 层：状态级解耦**
- draft 阶段产 `execution_plan`，写入 task.draft_json
- execute 阶段读 task.draft_json + user_feedback，构造 ResearchState
- 用户可以在 draft 和 execute 之间隔天才调用，task 状态自动保留

**第 3 层：成本级解耦**
- draft 阶段成本极低（几次 LLM call + 9 次搜索）
- execute 阶段成本高（30-60 次 LLM + 100+ 次搜索）
- 解耦让"花大钱"的事在"看清楚方向"之后才发生

### 高频追问

**Q1：解耦的代价是什么？多了一个状态管理层是不是更复杂？**

A：代价是**多了一个状态存储 + 工具数量翻倍**：
- 状态存储：用 SQLite 解决，几张表 + 几个 get/set 方法
- 工具数量：draft + execute 分别一个工具，对 AI 助手来说稍多但容易理解

收益：用户可以"先 draft，离线评估，第二天再 execute"。如果绑死，draft 完必须立刻 execute，用户体验受限。

整体 trade-off 值得。

**Q2：解耦之后如果 draft 跟 execute 之间用户改了配置（比如换了 LLM key），execute 还能跑吗？**

A：能跑，但有边界：
- draft 阶段的产物（execution_plan）是 LLM 生成的策略，跟 LLM 厂商无关
- execute 阶段重新用当前配置启 LLM，按策略执行
- 所以换 LLM key 没影响

**例外**：如果 draft 用的 model_a 跟 execute 用的 model_b 风格差距大（比如 a 是中文偏向、b 是英文偏向），写出来的报告风格可能不一致。但这是 edge case。

**Q3：如果 draft 阶段 LLM 生成的策略本身就有问题（比如让查不存在的来源），怎么发现？**

A：3 道防线：
1. **draft 后 LLM 输出 schema 校验**：planner.normalize_execution_plan 会修正格式问题（缺字段、类型错）
2. **HITL 用户审批**：用户看到策略不合理直接拒绝或修改
3. **execute 期间 supervisor 再加工**：把抽象的策略翻译成具体 SubTask 时，规则层会加默认值

但**深层语义问题**（比如策略本身错误方向）依赖 HITL 兜底。这就是 HITL 的价值。

---

## 4.3 「Writer 节点基于大纲与证据包进行分章节报告撰写」

### 一句话先讲清

Writer 不是看全部卡片就开写，而是看 outline（大纲）+ 每个 section 的 digest（证据包），一个 section 一个 section 地写。

### 关键术语

- **大纲（outline）**：报告的章节结构。由 outline_builder 节点在 reflector 之后生成。
- **证据包（section_digest）**：单 section 的卡片压缩版 + 元数据。
- **分章节撰写**：每个 section 独立调一次 LLM，最后拼接，跟"一次性写完全篇"对比。

### 详细解释

完整 writer 流程：

```
当 writer 被调用：

1. 拿到 plan_data（含 sections 列表）和 section_digests 字典

2. 对每个 section_plan 顺序处理：
   2.1 找出该 section 引用的 digest_ids
   2.2 合并多个 digest（如果一个 section 引用多个 digest）
   2.3 从所有 cards 里选 top 6-8 张 raw_cards（按 score + confidence + 关键词匹配）
   2.4 格式化 digest_for_writer + raw_cards_brief
   2.5 注入 conflict_block（如果该 section 有跨源矛盾）
   2.6 调 LLM 写本 section 的 markdown
   2.7 替换数字引用 [1] 为 <sup><a href="url">[1]</a></sup>

3. 调 LLM 写报告框架：introduction + direct_answer + future_outlook
4. 把所有 section 拼接，加 References 列表
5. 保存为 final_report.md
6. 同时输出 cards.json / sources.json / metadata.json / activity.json
```

### 高频追问

**Q1：为什么"分章节"而不是"一次性写完整报告"？**

A：4 个理由：
1. **上下文长度限制**：完整报告需要看所有证据（30-60 张卡片），prompt 太长，LLM 容易遗忘
2. **质量稳定性**：分章节让每次 LLM 调用聚焦——只看本 section 的证据写本 section，注意力集中
3. **失败隔离**：某个 section 写失败重试，不影响其他 section
4. **可并行潜力**：理论上 N 个 section 可以并行写（当前是串行，可优化点）

**Q2：每个 section 拿到的 raw_cards 是怎么选的？**

A：见 writer._select_raw_cards_for_section 函数：
1. 按 section_id / digest_ids 过滤（只要本 section 的卡片）
2. 按 evidence_requirements 关键词匹配加分
3. 按 evidence_score + confidence 排序
4. 取 top N（writer_raw_cards_per_section，默认 6）

**为什么不传所有卡片**：raw_cards 用于"精确引用"，要的是质量不是数量。6 张精选 > 30 张全量。

**Q3："基于大纲"——如果大纲跟证据不匹配怎么办？比如大纲提到一个章节但其实证据不足？**

A：3 层保护：
1. **大纲在证据之后生成**（详见 6.4 of project doc）：outline_builder 看 section_digests 才决定大纲，**不会写"想讲但没证据"的章节**
2. **writer 检查 digest_ids**：如果 section 引用的 digest_id 在 digest_map 里找不到，跳过该 section
3. **caveat 写入**：reflector 标的 critical_gaps 传给 writer，writer 在弱 section 加 caveat

**典型场景**：用户问"X 和 Y 对比"，但 Y 的资料严重不足。outline_builder 会调整大纲——可能只讲 X、或合并讲、或保留对比但加 limitations。

---

## 4.4 「采用封闭证据 ID 集 + verbatim quote / 数值容差校验的结构化 grounding 机制（对齐 Gemini Deep Research 设计）」

### 一句话先讲清

让 LLM 写报告时**物理上无法编造引用号**——给它一个固定的 ID 集合（E1, E2, ...），让它必须从中选，系统拿到结构化输出后再渲染引用号。

### 关键术语

- **封闭证据 ID 集（closed evidence ID set）**：给 LLM 一个明确的 ID 列表（E1..EN），LLM 只能引用这些。
- **verbatim quote**：LLM 引用数值类陈述时必须给出原文一字不差的句子作为 grounding。
- **数值容差校验**：段落里出现的数字必须能在引用的 excerpt 里找到（±5% 容差），找不到 → 标记 numeric_unverified。
- **结构化 grounding**：让 LLM 输出结构化 JSON 而不是自由 markdown，每个段落显式声明引用了哪些 ID。
- **对齐 Gemini Deep Research 设计**：Google Gemini 的 Deep Research 公开技术 blog 表明它用类似的"约束生成"思路。

### 详细解释

**完整 grounded generation 流程**：

```
当 writer 处理某一个 section：

1. 选出该 section 的 raw_cards（最多几张精选）

2. 给每张卡片打上短 ID：E1, E2, E3, ... EN
   同时记录每个 ID 对应的 reference_number（最终在报告里显示的引用号）

3. 构造 prompt 喂给 LLM：
   "你只能引用以下 evidence ID（不能编造其他 ID）：
   [E1] (ref 3) source: SiFive 官方
     excerpt: 'P870 achieves 95% of Cortex-A78's IPC...'
   [E2] (ref 4) source: Anandtech 评测
     excerpt: 'measured 87% in our internal tests...'

   规则：
   - 输出 JSON: {paragraphs: [{text, evidence_ids, quote?}]}
   - evidence_ids 只能从 [E1, E2, ..., EN] 选
   - 数字/百分比/日期/人名必须给 quote 字段（verbatim 引文）
   - text 里不要写 [1] 这种标记（系统自己渲染）"

4. LLM 返回 JSON

5. 系统验证（validate_paragraphs）：
   - parse JSON 失败 → 降级到老的 free-form 路径
   - 对每个段落：
     a) evidence_ids 检查闭集：不在 {E1..EN} 里的丢弃，记 invalid_ids_dropped
     b) 如果有 quote 字段：quote 的 token 必须全部在引用的 excerpt 的 token 里
        （token 级 containment，容忍标点和空白差异）
        不通过 → 记入 quote_failures（不删段落，只标记）
     c) 段落文字里有数字 → 必须在引用的 excerpt 里有 ±5% 容差的匹配
        不通过 → 记入 numeric_failures（不删段落，只标记）
   - 如果所有段落 evidence_ids 全失效 → 整个 section 降级到老路径

6. 系统渲染 Markdown：
   - 段落文本 + 末尾 [3][4]
   - LLM 写的 [99] 之类的字符串被剥离（系统才有引用权）
   - [3][4] 这种纯文本最后由 _replace_numbered_citations_with_links
     转换成 <sup><a href="url">[3]</a></sup>

7. 累计 citation_audit 数据
```

**4 个结构性约束**：

| 约束 | 防什么 |
|---|---|
| 封闭 ID 集 | LLM 编造引用号（说"根据 [7]"但根本没有 7） |
| Verbatim quote | LLM 改 paraphrase 改飘了（原文说 87%，LLM 写 95%） |
| 数值容差 ±5% | 数值幻觉（最致命也最常见） |
| 系统渲染引用号 | LLM 自由控制引用排版 |

### 高频追问

**Q1：为什么是"封闭 ID 集"而不是"告诉 LLM 别编造"？**

A：**指令式约束没用，结构性约束才有用**。
- 旧版本：prompt 里写"never invent citations"——LLM 仍然会编（尤其在 evidence 不够支持论点时）
- 新版本：给 LLM 一个 ID 集 [E1, E2, ..., EN]，让它输出 evidence_ids 字段——系统拿到后做闭集校验
- LLM 即使想编 E7，写出来 system 也会丢弃；不在 ID 集里的根本无法转成引用号

**关键洞察**：约束 LLM 不能靠 prompt 说服，要靠**结构性强制**。

**Q2：±5% 容差怎么定的？**

A：测试中的折中：
- 太严（0%）：95% vs 95.3% 报错——明显是小数舍入
- 太松（20%）：95% vs 80% 通过——真错漏掉
- 5%：能容忍合理舍入（95% vs 95.3% 通过），能拦下真错（95% vs 87% 报错）

**测量方法**：没做大规模 A/B。是 design parameter，依赖常识。

如果严格要做：构造一组"包含数值的卡片 + writer 输出对"数据集，人工标对错，调容差 0/1/3/5/10% 看 F1。我没做。

**Q3：JSON 解析失败 / 所有段落降级 → 走老路径，老路径还能保证质量吗？**

A：老路径是 commit 4 之前的版本：自由 markdown + 事后审计。质量不如 grounded 路径，但**仍保留**：
- 引用 URL 必须在 allowed_urls 里（不能完全编造）
- 引用渲染为 <sup><a href="url">[N]</a></sup>（用户看到的样式一致）

降级后 audit 字段会标 `writer_fallback=true`，用户在 get_research_result 输出里看到，可以选择 reuse follow_up_research 重做该 section。

**实际跑下来**，捕获到的降级率是 **DeepSeek V3 < 5%、GPT-4o < 2%**——主流 LLM 都能正确输出 JSON。降级是 known safety net，不是常态。

**Q4：跟 Gemini Deep Research 比，你这个有什么不一样的？**

A：相同点：核心机制（封闭 ID + 结构化输出 + 系统渲染引用）一样。

我的差异化：
1. **更细的数值校验**：Gemini 公开 blog 没提数值容差，我这是显式做了 ±5% 校验。这是数值幻觉的"最后一道闸"。
2. **verbatim quote 字段**：Gemini 的具体实现没公开，我这是 explicit 要求 LLM 给 verbatim quote，再做 token containment 校验。
3. **可降级 fallback**：Gemini SaaS 出错可能就报"研究失败"。我有 fallback 到老路径，**永远能出报告**。
4. **审计透明**：citation_audit 字段暴露给用户，他能看到引用质量统计。SaaS 通常不暴露这些。

诚实说：跟 Gemini 比，我**算法对齐但工程深度还差**，比如 Gemini 应该有更强的语义验证（不只是数值）。

---

## 4.5 「从生成端而非事后审计端消除引用幻觉」

### 一句话先讲清

旧版本是"先让 LLM 自由写，再检查引用对不对"（事后审计）。新版本是"约束 LLM 只能用合法 ID，本来就不让它写错"（生成端）。

### 关键术语

- **生成端（at generation time）**：在 LLM 写的过程中就约束。
- **事后审计端（post-hoc verification）**：LLM 写完之后再检查。
- **引用幻觉（citation hallucination）**：LLM 编造引用——说"根据 [3]"但 [3] 不存在或不支持该论断。

### 详细解释

**两种范式对比**：

| 维度 | 事后审计 | 生成端约束 |
|---|---|---|
| LLM 写引用 | 自由写 | 必须从封闭 ID 集选 |
| 检查时机 | 写完后 | 验证 + 渲染时 |
| 编造引用号 | 检测后替换为 plain text | 物理上不可能 |
| 数字飘移 | 难检测（要语义比较） | 强制 verbatim quote + 容差校验 |
| 额外 LLM 调用 | +1-2 次（LLM 验证器） | 0 |
| 鲁棒性 | 单点 | 失败降级 |

### 高频追问

**Q1：事后审计为什么会失败？具体一个例子。**

A：典型场景：LLM 写"根据 [3]，模型 X 准确率达到 95%"，但 [3] 是一篇关于 X 模型架构的论文，**完全没提准确率**。
- 事后审计能查："[3] 是不是有效 URL？" → 是
- 事后审计查不出："[3] 真的说了 95% 吗？"——需要语义比较，要 LLM 再调一次，贵且不准
- 用户看报告："好像有引用，应该可信"——上当

生成端约束：LLM 拿到的是 E3 的 exact_excerpt（架构论文），看到里面没有 "95%"，要 quote 95% 就 quote 不出来。系统校验数值容差时发现"95% 不在 cited excerpts 里"，直接标 numeric_failure。

**Q2：生成端约束完全消除幻觉吗？**

A：诚实回答：**消除了"引用号编造"和"数值飘移"两大类，但不消除所有幻觉**。
- 能消除：引用号编造（封闭集）、数值飘移（容差校验）、误引用（quote token 校验）
- **不能消除**：语义解读错误（excerpt 说"X is moderate"，LLM 写成"X is fast"）。这是 paraphrase 范畴，需要 entailment 模型才能查。

但**前两类是 deep research 产品最严重也最常见的幻觉**，消除它们已经显著提升信任度。

**Q3：用户怎么知道引用是被 grounding 过的？**

A：通过 citation_audit 字段。get_research_result 输出包含：

```
citation_audit:
  sections_grounded: 6/6        ← 全部 section 走 grounded 路径
  sections_fallback: 0          ← 没降级
  citations_total: 28           ← 全部 28 个引用经过封闭集校验
  invalid_ids_dropped: 0        ← LLM 没尝试编 ID
  ungrounded_paragraphs: 0
  quote_failures: 0
  numeric_failures: 1           ← 有 1 句数字没在 excerpt 出现，用户可重点 review
```

用户看一眼就知道这份报告的引用质量。

---

## 4.6 「实现证据来源链接可追溯」

### 一句话先讲清

报告里每个引用都是可点击的超链接，跳到原始网页；报告末尾还有完整 References 列表。

### 关键术语

- **来源链接可追溯**：每个论断都能点回原始 URL，跟"只写'根据某研究'但不给链接"对比。

### 详细解释

实现细节：

**段内引用样式**：
- LLM 写完段落，evidence_ids = ["E1", "E3"]
- 系统按 reference_number 映射到 `[1][3]`
- 替换为 `<sup><a href="https://...">[1]</a></sup><sup><a href="https://...">[3]</a></sup>`

**末尾 References 列表**：
- writer 拼接报告时加 "## References"
- 按 reference_number 顺序列出每个 source：
  ```
  ## References
  1. [Source Title](https://...)
  2. [Source Title](https://...)
  ...
  ```
- 用户在客户端（Cursor / Claude Desktop）的 Markdown 渲染器里看到的就是可点击列表

**Sources 元数据 JSON**：
- 同时产出 `DeepResearch_Sources_{task_id}.json`
- 每个 source 包含 reference_number / source_title / source_url / source_type / first_used_section
- 便于程序化处理

### 高频追问

**Q1：超链接渲染依赖客户端，如果客户端不支持 Markdown 怎么办？**

A：3 层兜底：
1. **Markdown 文件本身**：用户可以打开 Markdown 文件在 VS Code / Typora 等查看
2. **References 列表**：纯文本也能看 "1. Title (https://...)" 形式的引用
3. **Sources JSON**：程序化提取，没格式依赖

实际上所有主流 MCP 客户端都支持 Markdown，这个 fallback 几乎用不上。

**Q2：用户点击引用跳转后看到的网页跟报告里的论断是否对得上？**

A：核心机制：
- exact_excerpt 字段在卡片里保存"原文中的具体句子"
- writer 引用某个 evidence_id 时，excerpt 就是它引用的"证据原话"
- 用户点击链接到原网页，可以 Ctrl+F 搜 exact_excerpt 找到对应位置

**known limitation**：网页内容可能更新（原网页 2 个月后改了），exact_excerpt 可能找不到。这是 deep research 通病——证据快照只在抓取时为真。可以加 archive.org 自动归档作为缓解，目前没做。

**Q3：如果引用的源是 PDF / 论文，跳转后不是普通网页怎么办？**

A：source_kind 字段标注了类型（web / paper / pdf / repo / news...）。
- web / news：直接跳网页
- paper：链接通常是 arXiv 或 SemanticScholar 的论文页
- pdf：可能直接是 PDF URL（用户浏览器打开 PDF）
- repo：GitHub 仓库链接

所有都是可点击 URL，用户能跟到原始资料。

---

# 第 5 章：工程基础设施段

## 5.1 「通过 LiteLLM 统一适配层接入多模型供应商」

### 一句话先讲清

不直接调 OpenAI / Anthropic / DeepSeek 的 SDK，而是用 LiteLLM 一个库统一调，换厂商只改 1 个环境变量。

### 关键术语

- **LiteLLM**：开源 Python 库，把不同 LLM 厂商的调用接口统一。
- **统一适配层（unified adapter layer）**：业务代码不感知具体厂商，只对接一个统一接口。
- **多模型供应商**：DeepSeek、OpenAI、Anthropic、Google Gemini 等。

### 详细解释

代码层的体现：

所有 LLM 调用都走 `litellm.acompletion(model="...", messages=[...])`，model 字段是 `provider/model-name` 格式：
- `deepseek/deepseek-chat`
- `openai/gpt-4.1-mini`
- `anthropic/claude-3-5-sonnet`
- `gemini/gemini-pro`

LiteLLM 自动识别 provider 部分，调对应厂商 API。

**配置切换**：
- 环境变量 `DEEP_RESEARCH_LLM_MODEL` 指定用哪个模型
- 对应的 API key 用厂商专属 env（DEEPSEEK_API_KEY / OPENAI_API_KEY / ...）
- LiteLLM 内部读这些 env

**per-role 模型支持**：
- `DEEP_RESEARCH_PLANNER_MODEL` / `RESEARCHER_MODEL` / `WRITER_MODEL` / `REVIEWER_MODEL`
- 可以给不同角色用不同 LLM（比如 planner 用便宜的 gpt-4-mini，writer 用 claude）

### 高频追问

**Q1：为什么要套 LiteLLM 一层？直接用 OpenAI SDK + Anthropic SDK 不行吗？**

A：3 个理由：
1. **代码量节省**：每个厂商 SDK 接口不一样（OpenAI 用 `client.chat.completions.create`，Anthropic 用 `client.messages.create`），N 个厂商要写 N 套 wrapper。LiteLLM 一套搞定。
2. **切换成本**：换厂商时改 1 行 env，不动代码。直接用 SDK 要改业务代码。
3. **统一参数 / 错误**：不同厂商参数名差异（max_tokens vs max_output_tokens 等）由 LiteLLM 抹平。

**Q2：LiteLLM 是不是引入了新的依赖风险？如果它挂了怎么办？**

A：是有依赖风险，但权衡值得：
- LiteLLM 是活跃维护的项目（v1.50+，每周更新）
- 即使挂了，可以 fork 或下沉到直接 SDK 调用
- 业务代码里只用 `call_llm_text` / `call_llm_json` 两个函数（agents/base.py），切换成直接 SDK 改动只在这两个函数内

**Q3：per-role 模型支持有什么实际用途？**

A：成本优化场景：
- **Planner**（轻量推理 + JSON 输出）：用便宜的 gpt-4o-mini
- **Researcher**（同上）：同样轻量模型
- **Writer**（需要长文质量）：用更贵的 Claude Sonnet 或 GPT-4
- **Reviewer**（评估打分）：用 gpt-4o-mini

整体成本可降 30-50%（粗估，没做严格对比）。

---

## 5.2 「使用 SQLite 持久化图执行状态」

### 一句话先讲清

LangGraph 每个节点执行完都把当前 state 存到 SQLite 文件，**任务跑到一半挂了，重启能从最后一个 checkpoint 恢复**。

### 关键术语

- **图执行状态（graph state）**：LangGraph 跑过程中所有节点共享的数据（task_id、cards、sub_tasks 等）。
- **持久化**：写到磁盘，跨进程重启能读回。
- **SQLite**：嵌入式数据库，单文件存储。

### 详细解释

实际实现：

```
LangGraph 的 AsyncSqliteSaver 自动给每个节点执行完写一次 state 快照
存储位置：~/Desktop/DeepResearch/_runtime/DeepResearch_GraphCheckpoints.sqlite
表结构：thread_id (= task_id) + checkpoint_id + state_blob
```

**恢复场景**：
1. 进程崩了 → 重启 → 调 tool_check_status → 系统从最后 checkpoint 读 state → 继续
2. 用户追问 → tool_follow_up_research 从原 task_id 的 checkpoint 读 state → 复用之前的卡片 → 增量研究

### 高频追问

**Q1：为什么用 SQLite 不用 Redis 或 PostgreSQL？**

A：SQLite 的优势在 deep research 场景：
- **零运维**：单文件，不需要起服务
- **嵌入式**：跟主进程同一个 Python 进程读，没网络开销
- **足够快**：deep research 一个 task 的 checkpoint 写入 < 100KB，SQLite 完全够用
- **可便携**：备份/迁移就是拷一个文件

Redis 适合"高并发缓存"，这不是 deep research 的场景（一个 task 跑几分钟，QPS 极低）。
PostgreSQL 适合"多客户端高并发事务"，过度设计。

**Q2：SQLite 的 checkpoint 数据量会无限增长吗？**

A：会增长，所以做了 prune CLI：
- `deep-research-mcp prune` 默认删 30 天前的 terminal task（completed / failed / cancelled）
- 同时清理 task 注册表和 checkpoint SQLite
- 支持 `--dry-run` 预览

**Q3：如果 checkpoint SQLite 文件损坏怎么办？**

A：3 道防线：
1. **WAL 模式**：SQLite WAL 自带 crash recovery，进程崩了 WAL 文件能恢复
2. **任务隔离**：每个 task 在 SQLite 里是独立 thread_id，单 task 损坏不影响其他
3. **报告本身在 Markdown**：即使 checkpoint 全丢，已生成的报告还在 ~/Desktop/DeepResearch/ 目录下

但如果 SQLite 文件级损坏，只能丢弃未完成的 task。这是 known limitation。

---

## 5.3 「支持长时任务恢复与增量追问」

### 一句话先讲清

任务可以跑几十分钟，期间断了能恢复；用户做完研究后还可以追问，**不用重新搜索**已经查过的东西。

### 关键术语

- **长时任务**：deep research 一个任务 3-8 分钟，复杂的可能 30 分钟。
- **任务恢复**：进程崩了重启后能继续。
- **增量追问（incremental follow-up）**：基于已有报告补充研究。

### 详细解释

**长时任务恢复机制**：

```
当任务在运行中进程崩溃：

1. 客户端再次启动 deep-research-mcp
2. 用户调 tool_check_status(task_id)
3. service 读 SQLite，发现 task lifecycle = "running" 但进程没在跑
4. 实际上 LangGraph state 在 checkpoint 里
5. 系统标 task 为"interrupted"
6. 用户可以调 tool_execute_plan 让它继续
   - LangGraph 自动从最后 checkpoint 接上跑
```

**增量追问机制**：

```
当用户调 tool_follow_up_research(task_id, "再深挖一下 X 子话题"):

1. 系统从 checkpoint 读原 task 的 state（含 all knowledge_cards）
2. 调 planner.plan_follow_up：
   - 给 LLM 原 plan + 已有卡片 + follow_up_question
   - 让 LLM 决定：是"深挖某个已有 section"还是"加新 section"
   - 输出 new_sub_tasks（只针对增量部分）
3. 启动增量 graph 执行：
   - 只跑 new_sub_tasks
   - knowledge_cards 在原集合上 append
4. Writer 重写报告
5. 保存为 version 2
6. 用户可用 compare_report_versions 看 diff
```

**关键设计**：不重新搜原 section 的内容——复用已有卡片。

### 高频追问

**Q1：长时任务恢复有什么 edge case？**

A：3 个：
1. **正在执行 LLM 调用时崩溃**：当前 LLM 调用丢失，重启后从前一个 checkpoint（节点完成时）继续，会重复跑这个节点。**幂等性**靠业务保证——比如重复查同一个 query 拿到同样结果，结果合并不变。
2. **正在写文件时崩溃**：可能 partial write。但 cards.json / sources.json 都是 writer 节点最后一次性写，崩溃只丢这一次 write。next attempt 重写。
3. **SQLite 锁住**：另一个 deep-research-mcp 进程并发跑（多客户端同时调）。靠 WAL 模式 + busy_timeout 应对。

**Q2：增量追问怎么决定是"深挖"还是"新章节"？**

A：planner.plan_follow_up 让 LLM 判断：
- LLM 看到原 plan_data.sections + follow_up_question
- 输出 mode: "deepen_existing" / "add_section" / "cross_validate"
- 不同 mode 生成不同的 new_sub_tasks

实现上 LLM 决策有方差，但 plan_follow_up 输出的 new_sub_tasks 是结构化的，下游可以稳定执行。

**Q3：追问的 LLM 成本怎么样？跟重新做完整研究比省多少？**

A：典型场景：
- 完整研究：30-60 次 LLM
- 追问：5-15 次 LLM（只跑增量部分）

省 60-80% 的成本。**关键是复用了所有原 cards 而不是重新搜**。

---

## 5.4 「将向量嵌入与重排序剥离至独立 Worker 进程异步处理，避免主进程阻塞」

### 一句话先讲清

bge-embedder 和 bge-reranker 这两个本地模型加载慢、占内存，所以放到独立子进程里跑，主进程通过队列发任务，不卡。

### 关键术语

- **向量嵌入（embedding）**：把句子变成 384 维数字向量。
- **重排序（reranking）**：第二轮筛选，把召回的 top 100 用更精细的模型排出 top 10。
- **Worker 进程**：跟主进程隔离的子进程，专门跑模型。
- **异步处理**：主进程不等 worker 完成，先发任务、做别的事、稍后取结果。
- **避免主进程阻塞**：主进程的 asyncio 事件循环不被卡。

### 详细解释

**为什么不放主进程**：
- bge embedder 加载耗时 5-10s，期间 Python 主线程完全卡住
- asyncio 事件循环卡住 = 所有 MCP 工具调用都没响应

**子进程架构**：

```
主进程 (asyncio event loop)
  ├─ job_queue (mp.Queue, maxsize=64)  ← 限流防 OOM
  ├─ result_queue (mp.Queue)
  └─ worker_state (mp.Manager dict)    ← 共享心跳

子进程 (model_worker_process)
  ├─ 启动时加载 bge embedder + bge reranker + faiss
  ├─ heartbeat 线程（每 5s 更新 worker_state["heartbeat"]）
  └─ 主循环：从 job_queue 取任务 → 处理 → 写 result_queue

主进程调用方式：
1. asyncio.to_thread(job_queue.put, task)  ← 不阻塞 event loop
2. 轮询 result_queue 取结果（同样用 to_thread）
```

**worker 处理一个 job 的流程**：
```
1. 切片：每个文档按段落切，长段落进一步按句子切，每个 chunk 800 字符
2. 嵌入：用 bge embedder 把所有句子嵌入成向量
3. 建 FAISS 索引：内积索引（normalized）
4. 召回：对每个 query，找 top 8×max_results 个最相似句子
5. 重排：对每个 (query, parent_chunk) 用 cross-encoder 算精细分
6. 去重排序：按分数排，去重 (url, excerpt)
7. 返回 top N evidence
```

### 高频追问

**Q1：子进程通信比函数调用慢，为什么还要这么搞？**

A：trade-off：
- 函数调用：快，但模型加载和推理在主进程跑，**asyncio 全卡**
- 子进程通信：慢（多 1-10ms），但**主进程完全不卡**

deep research 一个 task 通常调 worker 10-20 次，每次多 10ms = 多 200ms 总延迟，**完全可接受**。换来的是主进程响应能力。

**Q2：worker 子进程挂了怎么办？**

A：4 道防线：
1. **心跳检测**：主进程每秒检查 worker_state["heartbeat"]，超过 300s 没更新 → 认为卡死
2. **自动重启**：标记 stale 后 stop + 重新 spawn
3. **重启滑动窗口预算**：5 分钟内最多 5 次重启，超额拒绝（避免无限重启循环）
4. **背压**：job_queue maxsize=64，满了主进程 put 会阻塞 → 自然限流

**Q3：子进程加载模型慢，启动开销是不是太大？**

A：是有开销（首次启动 30-60s），但**懒启动 + 长生命周期**：
- worker 只在第一次 tool_execute_plan 时启动
- 启动后驻留，后续 task 直接复用
- 服务关闭时一起关

对单 task 来说，启动开销在第 1 个任务摊销；后续任务零开销。如果用户跑多任务，整体效率很高。

---

## 5.5 「任务存储采用 aiosqlite + WAL 支持并发读写」

### 一句话先讲清

任务注册表用 aiosqlite（异步 SQLite）而不是同步 sqlite3，并开 WAL 模式让多个客户端能同时查询不卡。

### 关键术语

- **aiosqlite**：SQLite 的 async 版本，让 SQL 操作不阻塞 asyncio 事件循环。
- **WAL（Write-Ahead Logging）**：SQLite 的一种 journal 模式，**允许多个读 + 一个写并发**。

### 详细解释

**为什么不用同步 sqlite3**：
- 同步 `conn.execute()` 会阻塞调用它的协程
- 多个 MCP 工具并发调用时，一个 SQL 操作阻塞 → 其他工具等
- 改用 aiosqlite，SQL 操作放到 IO worker thread，主 event loop 不阻塞

**WAL 的价值**：
- 默认 DELETE 模式：读写互斥，写时所有读都等
- WAL 模式：写操作写到 WAL 文件，读操作仍读主文件，**互不阻塞**
- 适合本项目的场景：1 个写者（researcher 添加 events）+ N 个读者（status 工具被 N 个客户端查）

**WAL pragmas**：
```
PRAGMA journal_mode = WAL          ← 启用 WAL
PRAGMA synchronous = NORMAL        ← 性能优化（FULL 慢 2 倍，对本场景过度）
PRAGMA busy_timeout = 5000         ← 锁冲突时等 5s
```

### 高频追问

**Q1：什么场景下并发读写明显？**

A：3 个场景：
1. **多客户端同时挂同一 deep-research-mcp**：多个 Claude Desktop 实例配同一个 server（虽然 stdio 是 1:1，但理论上可以多 worker 模式）
2. **一个 task 在跑 + 另一个 task 查状态**：A 任务 researcher 正在 append events，B 任务 user 调 get_research_status
3. **后续追问场景**：原 task 已完成保存，但 SQLite 还在被新 task 写 → 用户查老 task 的报告

不开 WAL 的话，第 2/3 场景会有明显延迟。

**Q2：WAL 有什么 trade-off？**

A：2 个：
1. **多了一个 .wal 文件**：占额外磁盘空间，崩溃恢复时间略长
2. **写性能略降**：写要先写 WAL 再 checkpoint 到主文件（但本项目写量极小，无感知）

收益（读不阻塞写、写不阻塞读）远大于代价。

**Q3：aiosqlite 跟 sync sqlite3 性能对比？**

A：在 IO 密集场景 aiosqlite 整体吞吐更高（事件循环不阻塞）。单条 SQL 延迟略高（多了线程切换），但本项目场景下不敏感。

实测：100 个并发写在 0.03 秒内完成（详见 test_storage 的 concurrent test）。

---

## 5.6 「搜索层引入 per-engine 熔断 + 速率限制以隔离第三方故障」

### 一句话先讲清

每个搜索引擎独立有熔断器（连续失败就暂停 60s）和限流器（同时最多 N 个并发），不让一家 API 抽风影响全部。

### 关键术语

- **熔断（Circuit Breaker）**：电路保险丝同款。连续 N 次失败 → 自动断开 60s，期间所有调用直接返回空。
- **速率限制（Rate Limit）**：用 asyncio.Semaphore 限制并发数，每个引擎默认 4 个并发上限。
- **per-engine**：每个搜索引擎一个独立熔断器和限流器，互不影响。
- **第三方故障**：搜索 API 5xx、超时、限流。

### 详细解释

**熔断器状态机**：

```
CLOSED （初始状态）：
  - 正常调用
  - 成功 → 重置失败计数
  - 失败 → 失败计数 +1
  - 失败计数 ≥ 3 → 转 OPEN

OPEN （熔断状态）：
  - 直接返回空，不调用 API
  - 60s 后自动尝试一次
  - 成功 → 转 CLOSED
  - 失败 → 保持 OPEN，再等 60s
```

**限流器**：
- 每个 retriever 一个 asyncio.Semaphore(4)
- 同时只允许 4 个 task 调它
- 第 5 个 task 等前面 4 个之一结束

**配置**：
```
DEEP_RESEARCH_SEARCH_CB_THRESHOLD=3        ← 连续失败 3 次熔断
DEEP_RESEARCH_SEARCH_CB_COOLDOWN_SEC=60    ← 熔断 60s
DEEP_RESEARCH_SEARCH_RATE_LIMIT=4          ← 每引擎并发 4
```

### 高频追问

**Q1：熔断阈值 3 / cooldown 60s 怎么定的？**

A：经验值：
- 阈值 3：连续 3 次失败说明真出问题了（1-2 次可能是瞬时抖动）
- cooldown 60s：足够第三方恢复（典型 5xx 持续 30-180s）

测试时构造了模拟"flaky retriever"（每 3 次失败 1 次成功），调过参数感受效果。如果要严肃做需要根据真实第三方 API 的故障模式调（每家不一样）。

**Q2：限流 4 个并发是基于什么？**

A：保守值。理由：
- 大部分免费/低价层 API 限流是 10/s 左右
- 4 个并发 * 假设平均 RT 500ms = 8/s，留余量
- 用户花钱升级层级可以改 env 提到 8 或 16

**Q3：per-engine 隔离的意义？所有引擎共享熔断不行吗？**

A：差异很大：
- 共享熔断：一家挂了所有都不查 → 不必要的损失
- per-engine：Tavily 挂了只跳过 Tavily，Exa 还能查 → 系统整体可用性高

deep research 的核心价值之一就是**多源 fallback**。熔断必须独立。

---

# 第 6 章：高频综合性追问

## Q1：你做这个项目最难的点是什么？

我会答：**让引用真正可信**。其他子系统（搜索、并行、状态管理）相对模式化，难但有现成方案。引用幻觉是 deep research 类产品**最致命的信任问题**，市面上几乎所有产品（包括 Perplexity 早期）都被 reddit / hackernews 用户投诉过。

我的解法演化：
- 第一版：自由 markdown + 事后审计 URL → 不够
- 第二版（commit 4 redesigned）：**结构化 grounded generation**（封闭 ID 集 + verbatim quote + 数值容差）

具体技术挑战：
1. **怎么让 LLM 必须用 evidence_ids**：靠 prompt 指令没用，要靠**结构化输出 + 闭集校验**
2. **数值幻觉怎么抓**：regex 抓数字 + ±5% 容差比对引用 excerpt 里的数字
3. **失败怎么降级**：JSON 解析失败必须有 fallback 路径，否则任务会卡死

## Q2：你最自豪的设计是什么？

**结构化 grounded generation**（同上）。理由：
1. **这是产品质量的根本提升**，不是 incremental optimization
2. **设计灵感来自 Gemini Deep Research**，对齐了业界 SOTA
3. **改造跨多个层面**（writer prompt、citation_grounding 模块、graph 节点、tools 输出）但**对外行为完全兼容**（fallback 老路径）
4. **新增 25 个单元测试做回归保护**

## Q3：如果让你重做，会改什么？

3 个方向：

**优先级 1：评测体系**
- 构造 50+ 个 golden set 问题
- 跑系统产生报告
- 人工标注覆盖度、引用准确率、可读性
- 用这个 set 调所有参数（饱和度阈值、半衰期、数值容差）
- 当前所有阈值都是经验值，缺数据支撑

**优先级 2：流式输出**
- 当前 writer 是一次性产报告
- 可以改成"section 写完一段就推一段"
- 用户体感大幅改善（不用干等几分钟）

**优先级 3：嵌入语义去重**
- 当前是 MD5 哈希去重，同义不去重
- 可以加嵌入相似度去重作为可选项（同时保留多源 corroboration 的价值，需要小心设计）

## Q4：跟 Perplexity / Gemini Deep Research 比有什么差距？

诚实回答：

**算法层面**：核心机制（grounded generation、reflector loop）跟 Gemini 对齐，但细节深度不够（比如他们应该有更强的语义验证）。

**工程层面**：
- 我的是单机 / 单进程，他们是分布式系统
- 我的搜索源是用户自己接，他们有合作来源（学术数据库直连）
- 我的延迟跟 LLM 厂商挂钩，他们能调度多机器并行

**产品层面**：
- 我是工程师工具（MCP server），他们是消费级产品
- 我无 UI 美化，他们有专门设计的报告渲染

**优势**：
- 完全自部署，数据隐私可控
- 完全开源，工作流可改可调
- 完全免费（除了你自己的 API 费）

## Q5：你怎么测试这个系统的？覆盖率多少？

- 总单元测试 **119 个，0.85 秒跑完**
- 覆盖：
  - storage 层（aiosqlite + 并发）
  - knowledge_cache（去重）
  - 熔断器（CircuitBreaker）
  - 配置（validate_and_report）
  - 输入校验
  - 工具命令（init / doctor / prune）
  - retry/backoff（指数退避 + jitter）
  - 维护命令（prune）
  - recency 衰减（half-life）
  - 查询重写（多策略 + 容错）
  - 冲突检测（6 层过滤 + 严重度算法）
  - citation grounding（闭集 + quote + 数值容差）
  - writer 冲突 prompt 注入

诚实说**没有集成测试**——end-to-end 跑真实 LLM + 真实搜索的测试没做。理由：成本高（每次跑要花真 API 费）+ 测试不稳定（LLM 输出随机）。

替代方案：**手动 smoke test**——每次大改后手动跑一个真实研究任务，看输出质量。不严格但对个人项目够。

---

# 第 7 章：所有数字 / 参数 + 测量方法 + 基线

简历里出现的数字、参数全部解析。**测量方法 / 基线诚实标注**，没做严肃评测的地方就说没做。

| 简历数字 / 参数 | 实际值 | 测量方法 | 基线对照 | 调参依据 |
|---|---|---|---|---|
| 单次 LLM 调用产出 3 个备选 | 3 个 strategy | 设计参数（可配 1-5） | 旧版 1 个备选 | 经验：前 3 个策略覆盖最常见失败模式，更多边际递减 |
| 三层容错策略 | Tier 1/2/3 | 设计架构 | 旧版 2 层（retry + degrade） | 经验：3 层正好对应 transient / vocabulary / capacity 三类失败 |
| 4 个核心节点 | Supervisor / Researcher / Reflector / Writer | 设计架构 | 没分节点的话 monolithic | 按认知阶段划分：拆任务 / 干活 / 质检 / 写 |
| 饱和度阈值 0.85 | 0.85 | 经验值 | 旧版没饱和度 | 没严肃 evaluate；0.95 太严、0.7 太松 |
| 饱和度权重 60/40 | coverage 0.6 + marginal 0.4 | 经验值 | N/A | coverage 是直接信号、marginal 是辅助 |
| max_reflection_loops | 3 | 经验值 | N/A | 跑下来 90% 任务 1-2 轮停 |
| recency 半衰期（recent） | 6 个月 | 经验值 | 旧版无衰减 | 新闻类陈旧速度 |
| recency 半衰期（current） | 18 个月 | 经验值 | 旧版无衰减 | 大趋势类 |
| recency 半衰期（默认） | 36 个月 | 经验值 | 旧版无衰减 | 综合类 |
| recency floor | 0.3 | 经验值 | N/A | 极旧也不该完全抹掉 |
| 数值容差 ±5% | 5% | 经验值 | 旧版无校验 | 太严（0%）误报小数舍入，太松（20%）漏报真错 |
| 冲突检测 min_cards | 3 | 经验值 | N/A | < 3 卡片信号不足 |
| 冲突检测 max_pairs | 6 | 经验值 | N/A | 控制 LLM prompt 长度 |
| 冲突严重度阈值 | 0.7 strong / 0.4 moderate / < 0.4 weak | 经验值 | N/A | 跟 confidence 映射对齐 |
| 熔断阈值 | 3 次失败 | 经验值 | 旧版无熔断 | 1-2 次可能瞬时 |
| 熔断 cooldown | 60s | 经验值 | N/A | 第三方 5xx 典型 30-180s |
| 速率限制 | 每引擎 4 并发 | 经验值 | 旧版无限流 | 留余量给免费层限流 |
| Worker 队列 maxsize | 64 | 经验值 | 旧版无限制 | 防 OOM 同时不影响吞吐 |
| Worker 重启预算 | 5min 内 5 次 | 经验值 | 旧版无预算 | 防无限重启循环 |
| chunk_size | 800 字符 | 经验值 | N/A | bge embedder 输入上限 + 语义完整性折中 |
| chunk overlap | 150 字符 | 经验值 | N/A | 保证跨 chunk 句子能被某 chunk 完整覆盖 |
| writer_raw_cards_per_section | 6 张 | 经验值 | N/A | 上下文质量 vs 长度折中 |
| section_digest_max_cards | 24 张 | 经验值 | N/A | digest 信息完整性 |
| checkpoint 保留天数 | 30 天 | 经验值 | N/A | 用户回头看的典型时间 |
| 单元测试 | 119 个 | 实测 | 旧版 33 个 | 4 项功能升级各加 14-25 个 |

**面试关键话术**：
> "简历里这些数字大多是 design parameter 不是 measured improvement——意思是它们是经过 thought 的设计选择，但没有用 golden set 做严格 A/B 评测。这是项目 known limitation。"

诚实标注 known limitation 比假装做过评测安全。

---

# 第 8 章：自检 checklist（确认简历每句被解析）

把简历从头读一遍，逐句对应到文档章节。

## 段 1：项目介绍

- [x] 「构建面向开放域复杂问题的自主深度研究系统」→ **1.1**
- [x] 「通过 MCP 协议暴露标准化工具接口」→ **1.2**
- [x] 「支持 AI 助手即插即用调用」→ **1.3**
- [x] 「完成从调研执行策略确认到结构化报告的完整闭环」→ **1.4**

## 段 2：LangGraph 核心 Workflow

- [x] 「基于 LangGraph 构建以 Supervisor/Researcher/Reflector/Writer 为核心的 Workflow」→ **2.1**
- [x] 「采用 Map-Reduce 机制并行执行证据收集与结果汇总」→ **2.2**
- [x] 「Reflector 按研究任务评估证据覆盖度并生成补充检索任务」→ **2.3**
- [x] 「并结合跨轮次饱和度自适应路由至 Researcher 检索补强或 Writer 进行报告撰写」→ **2.4**

## 段 3：检索 + 证据中间层 + 时效 + 冲突

- [x] 「设计双层检索架构（多源发现/垂直检索）」→ **3.1**
- [x] 「在规划阶段引入多视角查询扩展」→ **3.2**
- [x] 「Researcher 层支持多策略查询重写（SIMPLIFY/SYNONYMS/DECOMPOSE/...，单次 LLM 调用产出 3 个备选）」→ **3.3**
- [x] 「三层容错策略保障局部失败不影响整体质量」→ **3.4**
- [x] 「构建证据中间层，将检索结果抽取为带置信度标注的知识卡片」→ **3.5**
- [x] 「全局去重后按任务压缩为章节证据包」→ **3.6**
- [x] 「为大纲生成和分章节写作提供高质量上下文」→ **3.7**
- [x] 「对时效敏感任务按 publication date 做半衰期衰减加权」→ **3.8**
- [x] 「section 内检测跨源矛盾并要求 Writer 显式呈现分歧而非静默仲裁」→ **3.9**

## 段 4：HITL + Grounded Citation

- [x] 「引入 Human-in-the-loop 审批机制」→ **4.1**
- [x] 「解耦调研执行策略与内容执行」→ **4.2**
- [x] 「Writer 节点基于大纲与证据包进行分章节报告撰写」→ **4.3**
- [x] 「采用封闭证据 ID 集 + verbatim quote / 数值容差校验的结构化 grounding 机制（对齐 Gemini Deep Research 设计）」→ **4.4**
- [x] 「从生成端而非事后审计端消除引用幻觉」→ **4.5**
- [x] 「实现证据来源链接可追溯」→ **4.6**

## 段 5：工程基础设施

- [x] 「通过 LiteLLM 统一适配层接入多模型供应商」→ **5.1**
- [x] 「使用 SQLite 持久化图执行状态」→ **5.2**
- [x] 「支持长时任务恢复与增量追问」→ **5.3**
- [x] 「将向量嵌入与重排序剥离至独立 Worker 进程异步处理，避免主进程阻塞」→ **5.4**
- [x] 「任务存储采用 aiosqlite + WAL 支持并发读写」→ **5.5**
- [x] 「搜索层引入 per-engine 熔断 + 速率限制以隔离第三方故障」→ **5.6**

**全部 29 句已解析。**

---

# 终极话术总结

如果面试官只让你讲 2 分钟，按这个版本：

> 我做了一个面向开放域复杂问题的深度研究系统，通过 MCP 协议接入 AI 助手。
>
> 核心是基于 LangGraph 的 Supervisor / Researcher / Reflector / Writer 四节点工作流，用 Send API 做 Map-Reduce 并行研究，Reflector 通过跨轮次饱和度评分自适应路由——简单问题 1 轮停，复杂问题最多 3 轮。
>
> 检索用通用层（Tavily/Exa 等 9 个引擎） + 垂直层（学术 SemanticScholar / arXiv / PubMed）的双层架构，配合多视角查询扩展、5 策略查询重写、3 层错误恢复来保证召回质量；
>
> 检索结果抽成带置信度的知识卡片做证据中间层，全局去重压缩成 SectionDigest，时效任务用 half-life 模型衰减老资料，section 内自动检测跨源矛盾让 writer 显式呈现分歧。
>
> 最大亮点是 Writer 的 grounded citation generation——给 LLM 封闭 evidence ID 集 + 要求 verbatim quote + 数值 ±5% 容差校验，对齐 Gemini Deep Research 的设计，从生成端消除引用幻觉。
>
> 工程上用 LiteLLM 接入多模型供应商，SQLite 持久化图状态支持长时任务恢复和增量追问，Worker 子进程跑 embedder/reranker 避免主进程阻塞，aiosqlite+WAL 支持并发读写，per-engine 熔断 + 限流隔离第三方故障。
>
> 整个项目 10000+ 行 Python，119 个单元测试。

文档结束。


File has not been read yet. Read it first before writing to it.