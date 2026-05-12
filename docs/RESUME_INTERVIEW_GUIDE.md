# Deep Research Engine 简历面试讲解指南

> 文档版本：v4.0  
> 适用项目：重构后的 Deep Research MCP Server  
> 目标：围绕最新简历描述，把每一句话都讲到“面试官能听懂、你也能接住追问”的程度。  
> 使用建议：不要死背术语。先用通俗话解释“为什么这样设计”，再补“代码里怎么实现”。

---

## 0. 最新简历描述

项目介绍：构建面向开放域复杂问题的自主深度研究系统，通过 MCP 协议暴露标准化工具接口，支持 AI 助手即插即用调用，完成从调研执行策略确认到结构化报告的完整闭环。

基于 LangGraph 构建以 Supervisor/Researcher/Reflector/Writer 为核心的 Workflow，采用 Map-Reduce 机制并行执行证据收集与结果汇总；Reflector 按研究任务评估证据覆盖度并生成补充检索任务，并结合跨轮次饱和度自适应路由至 Researcher 检索补强或 Writer 进行报告撰写。

设计双层检索架构（多源发现/垂直检索），在规划与检索任务生成阶段引入多视角查询扩展，检索链路配合三层容错策略保障局部失败不影响整体质量；构建证据中间层，将检索结果抽取为带置信度标注的知识卡片，全局去重后按任务压缩为章节证据包，为大纲生成和分章节写作提供高质量上下文。

引入 Human-in-the-loop 审批机制，解耦调研执行策略与内容执行；Writer 节点基于大纲与证据包进行分章节报告撰写，配合引用校验机制，实现证据来源链接可追溯，显著降低模型幻觉。

通过 LiteLLM 统一适配层接入多模型供应商；使用 SQLite 持久化图执行状态，支持长时任务恢复与增量追问。将向量嵌入与重排序剥离至独立 Worker 进程异步处理，避免主进程阻塞。

---

## 1. 一句话与整体讲法

### 一句话版本

这是一个本地运行的 Deep Research MCP Server。它不是让大模型直接凭记忆写报告，而是先让用户确认调研执行策略，再通过双层检索收集资料，把资料统一成 Document，再抽取 KnowledgeCard，压缩成 SectionDigest，基于证据生成大纲，最后按大纲和证据分章节写出带可点击来源的报告。

### 30 秒版本

这个项目解决的是复杂问题调研很难自动化的问题。普通搜索只能给链接，大模型直接写又容易幻觉。我把流程拆成了几个阶段：先根据用户问题生成调研执行策略并让用户确认；确认后并行搜索和抽取证据；Reflector 判断证据是否足够，不够就补查；足够后根据证据包生成大纲；Writer 再按大纲和证据包分章节写报告。最终正文里是可点击 [1] 小标，References 里是完整来源链接。

### 2 分钟版本

系统对外是 MCP Server，所以 AI 助手可以像调用工具一样调用它。内部用 LangGraph 编排，核心角色是 Supervisor、Researcher、Reflector、Writer。Supervisor 负责把调研执行策略变成研究任务；Researcher 负责双层检索、网页抓取、文档归一化、Worker 召回重排、LLM 抽取证据；Reflector 负责检查证据覆盖度和饱和度，决定是否补查；Writer 负责基于大纲、SectionDigest 和少量高相关 KnowledgeCard 写报告。

关键变化是：现在 draft 阶段不是报告目录，而是调研执行策略。原因很简单，没开始搜之前，系统其实不知道资料版图是什么，提前让模型凭先验写目录是不可靠的。所以 draft 会先生成 seed queries，做轻量侦察搜索，看有哪些来源类型、哪些方向可能有资料，然后输出 ResearchExecutionPlan 给用户确认。真正的大纲由 OutlineBuilder 在证据收集之后基于 SectionDigest 生成。

---

## 2. 先记住这 8 个核心判断

1. 用户确认的是调研执行策略，不是报告结构。
2. Map-Reduce 是按研究任务 / research track 并行，不是搜索前按报告章节并行。
3. 四个是核心角色，不代表代码里只有四个节点。
4. SearchService 是双层检索：普通搜索层负责广覆盖，垂直专业层负责专业增强。
5. 所有搜索和抓取结果都先统一成 Document，下游只吃稳定字段。
6. KnowledgeCard 是细粒度证据，SectionDigest 是按任务压缩后的证据包。
7. 大纲基于 SectionDigest 生成，并通过 evidence_digest_ids 和 evidence_requirements 约束 Writer。
8. Writer 按大纲写，每节使用对应 SectionDigest 和少量高相关原始 KnowledgeCard。

---

## 3. 简历第一段：项目定位与 MCP

> 构建面向开放域复杂问题的自主深度研究系统，通过 MCP 协议暴露标准化工具接口，支持 AI 助手即插即用调用，完成从调研执行策略确认到结构化报告的完整闭环。

### 30 秒通俗讲法

这个项目是一个“研究助手的后端大脑”。用户把复杂问题交给 AI 助手，AI 助手通过 MCP 调用我的工具。系统不会直接写报告，而是先生成一份调研执行策略，用户确认方向没问题后，再自动完成搜索、证据整理、大纲生成和报告写作。

### 2 分钟深入讲法

MCP 可以理解成给 AI 助手用的标准工具协议。项目提供了一组工具，覆盖研究任务的生命周期：

- `check_research_runtime`：检查模型、搜索配置、Worker、数据库是否可用；
- `draft_research_plan`：生成调研执行策略草稿；
- `start_research_task`：用户确认策略后启动正式研究；
- `get_research_status`：查询长任务进度和当前节点；
- `get_research_result`：拿到报告路径、预览和质量信息；
- `follow_up_research`：在已有研究基础上继续追问；
- `compare_report_versions`：比较不同轮次报告的差异。

这里的核心不是“多暴露几个 API”，而是把复杂研究拆成一个可控流程。draft 阶段先让用户确认方向，start 阶段才开始消耗搜索和模型资源。这样可以避免系统一开始理解错题，还能让用户补充时间范围、来源偏好、是否需要学术论文等信息。

### 面试官可能追问

**问：MCP 和普通 HTTP API 的区别是什么？**

推荐回答：HTTP API 主要给程序调用，调用方要知道接口、参数和返回结构。MCP 是面向 AI 助手的工具协议，工具描述会告诉模型这个工具能干什么、参数是什么，AI 助手可以根据用户自然语言自动选择工具。这个项目做成 MCP Server，是为了让 Claude Desktop、Cursor 或其他支持 MCP 的助手直接使用研究能力。

**问：为什么不是做一个 Web 页面？**

推荐回答：这个项目的重点是研究流程和证据链，不是前端产品。做成 MCP Server 可以复用现有 AI 助手作为交互界面，让用户直接在熟悉的对话环境里调用工具。后续如果需要 Web 页面，也可以在 MCP Server 外面再包一层 UI。

**问：完整闭环包括哪些环节？**

推荐回答：从用户问题开始，到策略草稿、人工确认、正式执行、并行检索、证据抽取、质量评估、补充检索、大纲生成、分章节写作、引用校验、报告落盘、状态查询、增量追问和版本对比，都在同一套任务体系里完成。

### 不能这么说

- 不要说“用户一提问，系统马上生成报告目录并开始写”。这会暴露对流程理解不准确。
- 不要说“MCP 就是 REST API”。它们都能暴露能力，但面向对象和调用方式不一样。
- 不要说“自动研究完全不需要用户参与”。项目明确引入了 Human-in-the-loop。

### 对应实现位置

- MCP 入口：`deep_research_mcp.py`
- 工具实现：`deep_research_runtime\tools.py`
- 服务组合：`deep_research_runtime\service.py`

---

## 4. 简历第二段：LangGraph、四个核心角色、Map-Reduce、Reflector

> 基于 LangGraph 构建以 Supervisor/Researcher/Reflector/Writer 为核心的 Workflow，采用 Map-Reduce 机制并行执行证据收集与结果汇总；Reflector 按研究任务评估证据覆盖度并生成补充检索任务，并结合跨轮次饱和度自适应路由至 Researcher 检索补强或 Writer 进行报告撰写。

### 30 秒通俗讲法

我把整个研究流程做成 LangGraph 状态机。可以把它理解成四个角色合作：Supervisor 分任务，Researcher 找资料和抽证据，Reflector 检查证据够不够，Writer 写报告。中间用 Map-Reduce 把多个研究任务并行执行，再统一合并结果。

### 2 分钟深入讲法

业务上说是四个核心角色，但工程上为了并行和汇总，会拆成更多图节点：

```text
supervisor
  -> dispatch_sections
  -> section_researcher x N
  -> collect_results
  -> reflector
  -> dispatch_sections 或 outline_builder
  -> writer
```

Supervisor 读取用户批准的 ResearchExecutionPlan，把它转成 SubTask。dispatch 节点把 pending 的研究任务并行派发给多个 section_researcher。每个 Researcher 独立执行检索和抽卡。collect 节点负责汇总和去重。Reflector 评估当前证据是否足以回答研究目标。如果发现关键缺口，就生成新的补充检索任务；如果证据足够或者新增信息已经饱和，就进入 outline_builder 和 writer。

这里的 Map-Reduce 不是传统离线大数据框架，而是一种工作流模式：Map 阶段并行执行多个研究任务，Reduce 阶段把证据、状态和质量信息合并回来。它并不是在搜索前就按最终报告目录并行，因为最终目录要等证据版图清楚后再生成。

### 面试官可能追问

**问：为什么选 LangGraph？**

推荐回答：这个任务是长流程、有循环、有状态、有并行、有恢复需求的。LangGraph 的图结构适合表达这些流程；checkpoint 可以保存执行状态；`Send()` 适合动态并行派发任务；reducer 可以处理多个并行节点同时写入状态的问题。

**问：四个核心角色和真实节点数不一致，怎么解释？**

推荐回答：四个核心角色是业务职责划分，真实图节点是工程实现拆分。例如 dispatch 和 collect 只是为了并行派发和结果收束，outline_builder 是为了把证据包变成大纲。它们是辅助节点，不影响“Supervisor / Researcher / Reflector / Writer”这条主线。

**问：Reflector 怎么知道还要不要继续搜？**

推荐回答：它会看证据覆盖度、每个研究任务的卡片数量、来源多样性、是否存在 critical gaps、是否有 degraded 任务影响关键结论，还会结合上一轮和这一轮信息增量。如果继续搜能补关键缺口，就回到 Researcher；如果已经饱和，就进入大纲和写作。

**问：路由为什么使用状态字段？**

推荐回答：路由决策写进状态后可以被 checkpoint 保存。长任务中如果刚做完评估但还没进入下一步就中断，恢复时可以读取已保存的路由结果，避免重新计算产生不一致。

### 不能这么说

- 不要把 Map-Reduce 说成“搜索前按最终报告章节并行”。现在是按研究任务并行。
- 不要说 Reflector 只是简单数卡片。它关注的是覆盖度、缺口和补查价值。
- 不要说四个核心角色等于代码中只有四个节点。

### 对应实现位置

- 图结构：`deep_research_runtime\graph.py`
- 状态模型：`deep_research_runtime\models.py`
- Reflector / Reviewer：`deep_research_runtime\agents\reviewer.py`

---

## 5. 简历第三段：双层检索、多视角查询、三层容错、证据中间层

> 设计双层检索架构（多源发现/垂直检索），在规划与检索任务生成阶段引入多视角查询扩展，检索链路配合三层容错策略保障局部失败不影响整体质量；构建证据中间层，将检索结果抽取为带置信度标注的知识卡片，全局去重后按任务压缩为章节证据包，为大纲生成和分章节写作提供高质量上下文。

### 30 秒通俗讲法

搜索这块不是简单调一个搜索 API。我把它分成两层：普通搜索层负责从 Web 上广泛发现资料，垂直专业层负责在需要时查论文这类专业来源。拿到结果后先统一成 Document，再抽成 KnowledgeCard，最后按任务压缩成 SectionDigest。这样 Writer 不直接面对杂乱网页，而是面对整理过的证据。

### 2 分钟深入讲法

SearchService 的结构可以这样讲：

```text
SearchService
  -> GeneralSearchLayer：Tavily / Exa / Serper / Bocha / DuckDuckGo fallback
  -> VerticalSearchLayer：academic 等专业插件
  -> Scraper / extractor：网页、Jina Reader、PDF 等内容提取
  -> DocumentNormalizer：统一字段
  -> Ranker / Worker：召回和重排
```

普通层是多源发现层，目标是找得广。垂直层不是普通搜索引擎的平替，而是专业增强层。比如用户要求学术论文、高质量期刊、会议、作者、年份、DOI，这时候 academic vertical 会被触发。这样既保留普通 Web 的覆盖面，也不会把学术源和普通搜索源混成一堆。

所有结果都会归一成 Document，关键字段包括：

- `url`：必须能让用户点回来源；
- `title`：来源标题；
- `content`：下游统一消费的正文；
- `source_layer`：general 或 vertical；
- `source_kind`：web、paper、pdf、news 等；
- `pdf_url`、`metadata`、`score` 等补充信息。

容错方面有三层：

1. retry/backoff：搜索 API 临时失败就重试；
2. query reformulation：结果太少或失败时改写查询；
3. degraded：还是不行就降级标记，交给 Reflector 判断是否必须补查。

证据中间层则是：

```text
Document -> KnowledgeCard -> SectionDigest -> Outline -> Report
```

KnowledgeCard 是单条证据，包含 claim、summary、exact_excerpt、source、confidence。SectionDigest 是按研究任务压缩后的证据包，用更大的卡片预算保留广度，主要服务大纲生成和章节写作的上下文控制。

### 面试官可能追问

**问：为什么要统一 Document？**

推荐回答：不同搜索源返回字段差异很大。有的叫 content，有的叫 raw_content，有的只有 snippet。如果下游直接依赖原始字段，接线会非常脆弱。统一 Document 后，Researcher、Worker、Writer 都只依赖稳定协议。

**问：普通搜索层和垂直层有什么区别？**

推荐回答：普通搜索层像“在全网找线索”，垂直层像“去专业数据库补强”。普通层适合新闻、公司页面、博客、文档；垂直层适合论文、医学、法规、专利等有专业元数据的场景。当前实现重点是 academic vertical，后续可以扩展其他领域插件。

**问：多视角查询扩展有什么用？**

推荐回答：复杂问题不是一个 query 能覆盖的。比如调研一个技术，可能要从原理、应用、指标、局限、最新进展、反面观点多个角度搜。多视角查询扩展就是让检索任务不要只沿着用户原句走，而是从多个研究视角构造查询。

**问：第三层 degraded 后是不是就不管了？**

推荐回答：不是。degraded 的意思是这个子任务当前轮没有拿到足够证据，但它的影响要交给 Reflector 判断。如果它影响主结论，Reflector 会生成补充检索任务；如果只是边缘信息，就不会为了它无限循环。

### 不能这么说

- 不要说“所有搜索源平铺在一个 retriever 列表里”。现在强调双层策略。
- 不要说“学术搜索就是普通搜索引擎之一”。academic 是垂直专业层。
- 不要说“降级后就放弃”。降级后还要经过 Reflector 判断影响。

### 对应实现位置

- SearchService：`deep_research_runtime\search_service.py`
- Document / KnowledgeCard / SectionDigest：`deep_research_runtime\models.py`
- Researcher：`deep_research_runtime\agents\researcher.py`
- 质量、去重、证据包：`deep_research_runtime\quality.py`

---

## 6. 简历第四段：Human-in-the-loop、大纲、Writer、引用

> 引入 Human-in-the-loop 审批机制，解耦调研执行策略与内容执行；Writer 节点基于大纲与证据包进行分章节报告撰写，配合引用校验机制，实现证据来源链接可追溯，显著降低模型幻觉。

### 30 秒通俗讲法

Human-in-the-loop 的作用是防止系统在理解错方向时直接开跑。用户先确认调研执行策略，然后系统执行研究。研究完成后，OutlineBuilder 根据 SectionDigest 生成大纲，Writer 再按大纲分章节写作。正文里保留可点击 [1] 小标，References 里显示完整标题和链接。

### 2 分钟深入讲法

这段最容易被追问，因为里面有“大纲”和“证据”的关系。

现在的大纲不是在搜索前生成的，而是基于 SectionDigest 生成的。SectionDigest 是从较多 KnowledgeCard 压缩出来的，它像一个“广度地图”，能告诉系统这个主题实际搜到了哪些方向、哪些来源、哪些结论、哪些缺口。OutlineBuilder 看这个证据地图，再决定最终报告应该怎么组织。

OutlineBuilder 输出的每个章节会包含：

- `title`：章节标题；
- `purpose`：本节写作目的；
- `questions`：本节要回答的问题；
- `evidence_digest_ids`：本节绑定哪些 SectionDigest；
- `evidence_requirements`：本节写作时需要优先满足的证据要求。

Writer 写作时不是只看一份压缩摘要。它会按大纲找到对应的 SectionDigest，然后再根据 evidence_requirements 选择少量更相关的原始 KnowledgeCard。这样做的好处是：SectionDigest 提供广度和结构，原始卡片提供精确摘录、来源、引用和细节。

引用上，正文保留学术报告常见的小标样式，但小标本身可点击：

```html
正文里：某个结论来自某项研究<sup><a href="https://example.com/source">[1]</a></sup>
References：1. [来源完整标题](https://example.com/source)
```

### 面试官可能追问

**问：为什么用户确认的是调研执行策略，而不是大纲？**

推荐回答：没开始检索前，系统不知道真实资料会集中在哪些方向。此时让模型生成大纲，很大程度是在用模型先验猜结构。更稳的做法是先确认搜索路径、来源范围、筛选标准和抽取字段，等证据收集后再基于证据生成大纲。

**问：大纲到底有什么用？**

推荐回答：大纲负责约束报告结构。它告诉 Writer 每一节写什么、回答哪些问题、使用哪些 SectionDigest，以及应该优先寻找什么类型的原始证据。没有大纲，Writer 容易把材料堆成摘要；有大纲，报告会更像结构化研究报告。

**问：evidence_digest_ids 是什么？**

推荐回答：它是章节和证据包之间的绑定关系。例如第二章需要解释技术路线，就绑定“技术原理”和“实现方法”两个 SectionDigest。Writer 写这一章时主要从这些证据包取材。

**问：evidence_requirements 是什么？**

推荐回答：它不是来源链接，也不是具体卡片 ID，而是取材要求。比如“需要包含实验指标”“需要覆盖反面观点”“优先使用 2023 年后的来源”。Writer 会用它来筛选原始 KnowledgeCard。

**问：为什么 Writer 还要少量原始 KnowledgeCard？**

推荐回答：SectionDigest 是压缩后的，适合提供全局理解，但写具体句子和引用时需要精确摘录、source、confidence 等信息。少量原始卡片能补足精确度，同时不会把上下文撑爆。

**问：引用校验具体校验什么？**

推荐回答：校验引用是否能回到 source catalog；正文引用是否有对应 URL；References 是否去重；是否出现孤立编号；URL 是否来自真实来源记录。它的目标是让用户能点击回源，而不是让编号只停留在文本形式上。

### 不能这么说

- 不要说“正文直接显示论文标题链接”。正文应该是可点击 [1] 小标。
- 不要说“大纲阶段指定所有原始卡片”。大纲绑定证据包并提出证据需求，具体卡片由 Writer 筛。
- 不要说“只要有引用就没有幻觉”。引用能显著降低风险，但还需要证据抽取和校验配合。

### 对应实现位置

- 策略草稿与大纲：`deep_research_runtime\agents\planner.py`
- Writer：`deep_research_runtime\agents\writer.py`
- 引用和来源目录：`deep_research_runtime\quality.py`
- 模型字段：`deep_research_runtime\models.py`

---

## 7. 简历第五段：LiteLLM、SQLite、Worker

> 通过 LiteLLM 统一适配层接入多模型供应商；使用 SQLite 持久化图执行状态，支持长时任务恢复与增量追问。将向量嵌入与重排序剥离至独立 Worker 进程异步处理，避免主进程阻塞。

### 30 秒通俗讲法

系统不绑定某一个大模型，而是通过 LiteLLM 统一调用不同供应商。任务状态存在 SQLite 里，所以长任务可以查状态、恢复，也能在已有报告基础上继续追问。embedding、FAISS 召回和 CrossEncoder rerank 这些重计算放到独立 Worker 进程里，主进程继续响应 MCP 请求。

### 2 分钟深入讲法

LiteLLM 的价值是统一模型调用接口。不同角色可以用不同模型，例如 Planner 需要规划能力，Researcher 需要稳定抽取 JSON，Writer 需要长文本写作，Reviewer 需要判断证据缺口。通过 role-based model override，可以在配置层调整，而不是在业务代码里写死某家模型。

SQLite 承担两个作用：

- task registry：记录任务状态、草稿、事件、报告版本、结果路径；
- LangGraph checkpoint：保存图执行状态，支持任务恢复。

Worker 独立进程处理向量相关任务：

- embedding：把文档块向量化；
- FAISS：做初步召回；
- CrossEncoder rerank：对候选证据重新排序。

这些任务比较耗时，如果放在 MCP 主进程，会让 status、result 这类工具调用也被卡住。拆到 Worker 后，主进程只负责调度和通信。

### 面试官可能追问

**问：为什么 SQLite 就够了？**

推荐回答：当前是本地 MCP Server，不是多租户高并发 SaaS。SQLite 部署简单、可靠、无额外服务依赖，适合保存本地任务状态。后续如果要做服务端多用户，可以替换成 PostgreSQL。

**问：增量追问怎么做？**

推荐回答：follow-up 会读取已有任务状态和报告版本，在已有证据基础上生成新的补充任务。执行完后产出新版本报告，并可以通过版本比较工具看差异。

**问：Worker 异步处理是不是会丢结果？**

推荐回答：主进程和 Worker 有任务队列、结果队列和超时处理。Worker 异常时任务会被标记 degraded，系统不会因为单个召回失败直接崩。

### 不能这么说

- 不要说“LiteLLM 让模型效果都一样”。它只是统一调用，不改变模型能力。
- 不要说“SQLite 适合无限并发”。当前定位是本地长任务持久化。
- 不要说“Worker 只是优化速度”。它也隔离了重计算，保证主进程可响应。

### 对应实现位置

- LLM 基类：`deep_research_runtime\agents\base.py`
- Runtime 与 Worker 生命周期：`deep_research_runtime\runtime.py`
- Worker：`deep_research_runtime\worker.py`
- 存储：`deep_research_runtime\storage.py`

---

## 8. 高频深挖专题

### 8.1 为什么 draft 阶段要做轻量侦察搜索？

通俗讲，用户的问题通常很短，但真正研究需要知道“资料在哪里”。例如用户说“调研某个技术的发展”，系统不能凭空决定报告结构。轻量侦察搜索的作用是先摸一下来源版图：有没有论文、有没有官方文档、有没有近期新闻、有没有权威报告。摸清楚以后，再生成调研执行策略给用户确认。

推荐回答：

> draft 阶段主要做方向确认，而不是正式研究。它会生成 seed queries，做少量侦察搜索，形成 source landscape，然后输出 ResearchExecutionPlan。这样用户确认的是搜索路径、筛选标准和信息抽取字段，而不是一个凭模型先验生成的报告目录。

### 8.2 为什么大纲要基于 SectionDigest？

因为全量 KnowledgeCard 可能很多，直接塞给模型容易超上下文，也会让模型被局部细节带偏。SectionDigest 用更大的卡片预算把每个研究任务的主要证据压缩成“广度摘要”，适合给 OutlineBuilder 生成大纲。

推荐回答：

> SectionDigest 相当于证据地图。它不是最终文本，但能告诉模型哪些方向证据多、哪些方向证据少、哪些来源可信、哪些结论反复出现。大纲基于它生成，比搜索前凭空生成更稳。

### 8.3 为什么 Writer 既看 SectionDigest 又看少量原始 KnowledgeCard？

SectionDigest 适合把握方向，但引用和细节需要原始卡片。原始 KnowledgeCard 有 exact_excerpt、source、confidence，可以支撑具体句子和引用。

推荐回答：

> SectionDigest 解决“看全局”的问题，原始 KnowledgeCard 解决“写准确”的问题。Writer 每节只取少量高相关卡片，是为了平衡上下文成本和证据精度。

### 8.4 大纲阶段是否要指定具体卡片或链接？

一般不应该。大纲阶段应该指定证据包和证据需求，而不是把具体卡片提前固定死。因为写作时才知道某一节需要哪些细节、哪些摘录最合适。

推荐回答：

> 大纲通过 evidence_digest_ids 绑定证据包，通过 evidence_requirements 表达取材要求。具体 KnowledgeCard 由 Writer 根据章节目标、证据需求、相关性、置信度和摘录质量筛选。

### 8.5 正文可点击 [1] 小标和 References 有什么区别？

正文小标是阅读时的引用入口，点击可以直接打开来源。References 是完整来源列表，显示标题、链接、来源信息，方便用户系统查看。

推荐回答：

> 正文保持学术报告常见的 [1] 小标，但 [1] 本身是链接。References 再展示完整标题链接。这样正文不被长标题打断，同时又保证来源能点击。

### 8.6 四个核心角色和辅助节点怎么讲不矛盾？

可以用“组织架构”和“执行细节”的比喻。公司里有产品、研发、审核、写作四类职责，但工程执行时可能有派单系统、汇总系统、模板系统。

推荐回答：

> Supervisor、Researcher、Reflector、Writer 是核心职责。dispatch、collect、outline_builder 是为了并行、汇总和证据驱动大纲生成而拆出来的工程节点。

---

## 9. 最容易出错的问答

### Q1：这个项目是不是就是搜索后总结？

不是。搜索只是第一步。关键在于策略确认、双层检索、Document 统一协议、证据抽取、去重、SectionDigest 压缩、Reflector 补查、大纲约束、分章节写作和引用校验。没有这些中间层，才是简单搜索总结。

### Q2：如果模型抽取的 KnowledgeCard 本身错了怎么办？

系统通过 exact_excerpt、source、confidence 和引用回溯降低风险。Writer 不应该只看 claim，而要结合摘录和来源。引用校验也要求 URL 来自 source catalog。它不能从理论上保证所有抽取都完美，但比直接让模型写报告可控得多。

### Q3：如果某个来源打不开，引用还有效吗？

最低要求是引用 URL 来自 source catalog，并且在生成时是可回溯的。如果未来网页失效，那属于外部来源生命周期问题。可以通过保存原始 excerpt、metadata、PDF URL 或快照来增强可复现性。

### Q4：为什么不用全量卡片写整篇报告？

全量卡片可能超上下文，而且会让模型在写作时难以聚焦。项目采用分章节写作：每节读对应 SectionDigest 和少量高相关卡片。这样既能利用大上下文模型，也避免上下文被无关证据污染。

### Q5：为什么不让大模型自己判断哪些来源可信？

可以让模型参与判断，但不能只靠模型。系统会保留来源类型、source_layer、score、confidence 等结构化字段，并在检索、抽卡、去重和写作阶段多次利用这些信号。

### Q6：为什么要有 source catalog？

source catalog 是引用可信度的基础。没有它，Writer 输出的 [1] 可能只是一个文本编号。source catalog 让每个编号能映射到真实 URL、标题和来源信息。

---

## 10. 推荐表达与高危表达对照

| 场景 | 推荐表达 | 高危表达 |
| --- | --- | --- |
| draft 阶段 | 生成调研执行策略，并通过轻量侦察搜索确认来源版图 | 一开始就让模型列报告目录 |
| 用户确认 | 用户确认搜索路径、筛选标准、抽取字段和不确定点 | 用户确认最终报告结构 |
| Map-Reduce | 按研究任务并行收集证据，再统一汇总 | 搜索前按最终章节并行 |
| 搜索架构 | SearchService 分普通搜索层和垂直专业层 | 所有搜索源放一个平铺列表 |
| 学术搜索 | academic 是 vertical plugin | academic 是普通搜索引擎之一 |
| 证据链路 | Document -> KnowledgeCard -> SectionDigest -> Outline -> Report | 搜索结果直接给 Writer |
| 大纲 | 大纲基于 SectionDigest，绑定 evidence_digest_ids 并给出 evidence_requirements | 大纲凭模型先验生成 |
| Writer | Writer 使用大纲、SectionDigest 和少量原始 KnowledgeCard | Writer 只看摘要或只看所有卡片 |
| 引用 | 正文可点击 [1] 小标，References 展示完整标题链接 | 正文塞入长标题链接 |
| 可靠性 | 证据来源链接可追溯，降低幻觉 | 保证完全没有幻觉 |

---

## 11. 代码地图速查

| 知识点 | 文件 |
| --- | --- |
| MCP 工具注册 | `deep_research_mcp.py` |
| 工具生命周期 | `deep_research_runtime\tools.py` |
| 服务组合 | `deep_research_runtime\service.py` |
| LangGraph 流程 | `deep_research_runtime\graph.py` |
| 状态与数据模型 | `deep_research_runtime\models.py` |
| Planner / OutlineBuilder | `deep_research_runtime\agents\planner.py` |
| Researcher | `deep_research_runtime\agents\researcher.py` |
| Reflector / Reviewer | `deep_research_runtime\agents\reviewer.py` |
| Writer | `deep_research_runtime\agents\writer.py` |
| SearchService | `deep_research_runtime\search_service.py` |
| 证据质量与来源目录 | `deep_research_runtime\quality.py` |
| Worker | `deep_research_runtime\worker.py` |
| 存储 | `deep_research_runtime\storage.py` |
| 配置 | `deep_research_runtime\config.py` |

---

## 12. 面试收尾模板

如果面试官让你总结项目，可以这样说：

> 这个项目最核心的价值，是把“复杂调研”从一次性大模型生成，改造成一条可控的证据流水线。它先确认调研执行策略，再通过双层检索收集资料，把资料统一成 Document，抽取 KnowledgeCard，压缩成 SectionDigest，基于证据生成大纲，最后按大纲和证据分章节写报告。整个过程有 Reflector 补查、三层容错、SQLite 持久化、Worker 异步召回重排和可点击引用，所以它更像一个工程化的研究系统，而不是简单的搜索总结脚本。
