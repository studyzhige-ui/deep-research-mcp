# Deep Research Engine 面试 QA 手册

> 文档版本：v4.0  
> 适用项目：重构后的 Deep Research MCP Server  
> 目标：覆盖项目定位、MCP 工具、LangGraph 流程、双层检索、证据链路、大纲、写作、引用、Worker、容错、持久化与扩展性。  
> 阅读方式：每个回答先讲“人话版”，再补实现细节。面试时先讲清楚设计动机，再说代码位置。

---

## 1. 项目定位与产品价值

### Q1：一句话介绍这个项目？

这是一个面向开放域复杂问题的自主深度研究系统。用户提出研究问题后，它先生成调研执行策略并让用户确认，然后自动完成检索、证据抽取、证据评审、补充检索、大纲生成和结构化报告写作，最终输出带可点击来源引用的 Markdown 报告。

实现上，它是一个 MCP Server，核心工作流由 LangGraph 编排，检索由 SearchService 统一管理，证据通过 Document、KnowledgeCard、SectionDigest 逐层整理，Writer 最后基于大纲和证据包分章节写作。

### Q2：它和“搜索引擎 + 大模型总结”有什么区别？

搜索引擎只解决“找链接”，大模型总结只解决“把已有文本写成答案”。这个项目真正做的是中间的研究流程工程化：

1. 先确认调研执行策略，避免一开始跑偏；
2. 用普通搜索层和垂直专业层扩大来源覆盖；
3. 把不同来源统一成 Document；
4. 抽取成带置信度和来源的 KnowledgeCard；
5. 全局去重后压缩成 SectionDigest；
6. Reflector 判断证据是否够，不够就补查；
7. 证据足够后生成大纲；
8. Writer 按大纲、SectionDigest 和少量原始卡片写报告；
9. 引用必须能回到 source catalog。

所以搜索和 LLM 只是底层能力，项目价值在于流程、状态、证据和质量控制。

### Q3：为什么叫“开放域复杂问题”？

开放域表示问题不限定在一个固定数据库里，可能涉及论文、新闻、公司文档、法规、技术博客、PDF、官方报告等多种来源。复杂问题表示它通常不是一个简单事实问答，而是需要多角度分析，比如“技术路线是什么、有哪些代表性论文、各自解决什么问题、未来趋势如何”。

这类问题如果只搜一次，覆盖面不够；如果只让模型回答，容易依赖模型先验。项目用研究任务拆分、并行检索、补查和证据压缩来处理复杂度。

### Q4：这个项目最核心的价值是什么？

最核心的价值是把“从问题到报告”的研究过程变成一条可控制、可恢复、可追溯的流水线。它不是一次性生成，而是逐步推进：

```text
用户问题
  -> 调研执行策略
  -> 用户确认
  -> 证据收集
  -> 证据评审
  -> 补查或进入大纲
  -> 分章节写作
  -> 引用校验
  -> 报告与版本保存
```

这个流程能减少两个风险：一是调研方向错，二是报告没有真实证据支撑。

### Q5：为什么不直接让大模型写报告？

因为复杂研究的问题通常需要事实来源。大模型直接写，可能写得很流畅，但来源不一定真实，结构也可能被模型先验影响。项目选择先收集证据，再让模型写作。模型仍然参与规划、抽取、评审和写作，但每一步都被结构化数据和来源约束。

简单说：模型负责理解和表达，系统负责证据流和流程控制。

### Q6：项目适合哪些场景？

适合需要较长调研、跨来源综合、需要引用的任务。例如：

- 学术论文综述；
- 技术方案调研；
- 行业报告；
- 产品竞品分析；
- 政策法规资料整理；
- 某个开源项目或技术生态的研究；
- 已有报告基础上的增量追问。

不太适合只需要一个简单答案的问题，比如“今天星期几”或“Python 怎么排序列表”。

---

## 2. MCP 与工具设计

### Q7：为什么做成 MCP Server？

MCP 是给 AI 助手调用工具的标准协议。做成 MCP Server 后，用户可以在支持 MCP 的 AI 助手里直接调用研究能力，不需要打开额外系统，也不需要自己写 API 请求。

从工程角度，它把交互界面和研究引擎解耦：AI 助手负责和用户对话，Deep Research Server 负责执行研究任务。

### Q8：MCP 层提供哪些工具？

主要工具包括：

- `check_research_runtime`：检查运行环境；
- `draft_research_plan`：生成调研执行策略草稿；
- `start_research_task`：用户确认后启动正式研究；
- `get_research_status`：查询任务状态；
- `get_research_result`：获取报告结果；
- `follow_up_research`：基于已有任务继续追问；
- `compare_report_versions`：比较不同版本报告。

这些工具形成了从策略确认到报告结果，再到后续追问的完整生命周期。

### Q9：为什么要有 `check_research_runtime`？

深度研究依赖模型、搜索 API、Worker、数据库和本地目录。任何一个环节不可用，都可能导致正式任务失败。`check_research_runtime` 的作用是提前发现问题，比如缺少搜索 key、Worker 启动失败、SQLite 路径不可写等。

这相当于正式跑长任务前做一次健康检查。

### Q10：`draft_research_plan` 到底输出什么？

它输出的是 ResearchExecutionPlan，也就是调研执行策略，不是最终报告目录。

它会说明：

- 系统怎么理解用户目标；
- 准备采用哪些搜索路径；
- 普通搜索层怎么搜；
- 是否需要垂直专业层；
- 筛选来源的标准；
- 要抽取哪些信息字段；
- 如何判断可信度；
- 哪些地方还不确定，需要用户确认。

### Q11：为什么 `start_research_task` 要和 draft 分开？

因为正式研究可能耗时、耗搜索额度、耗模型调用。如果 draft 阶段系统理解错了，直接开跑会浪费资源。分成 draft 和 start 后，用户可以在中间修正方向，比如增加“只要 2020 年之后的论文”或“优先看顶会顶刊”。

### Q12：状态查询为什么重要？

深度研究不是毫秒级请求，可能运行几分钟甚至更久。用户需要知道任务是否还在跑、当前处于哪个阶段、已经完成多少轮、是否遇到降级。`get_research_status` 就是为长任务交互设计的。

### Q13：增量追问是什么？

增量追问是指用户拿到报告后继续问：“再补充某个方向”“加入最新论文”“对比两个方案”。系统不需要完全从零开始，而是可以读取已有任务状态、已有证据和报告版本，生成补充任务，再输出新版本报告。

### Q14：版本比较有什么用？

增量追问会产生多个报告版本。版本比较可以告诉用户新版本相比旧版本增加了什么、修改了什么、哪些章节变化较大。相比纯文本 diff，它更关注语义变化。

---

## 3. Draft 策略与 Human-in-the-loop

### Q15：Human-in-the-loop 在这个项目里是什么意思？

意思是正式研究前必须有用户确认。系统先生成调研执行策略，用户看完后确认或修改，然后才启动正式研究。

这不是让用户参与每一步细节，而是在最关键的方向选择处让用户把关。

### Q16：为什么用户确认的是调研执行策略，而不是报告大纲？

因为没开始检索之前，系统还不知道真实资料版图。提前生成报告大纲，很容易变成模型基于已有知识和惯性结构猜出来的目录。更合理的是先确认“怎么查”：查哪些来源、查什么时间范围、用什么筛选标准、提取哪些字段。

报告大纲应该等证据收集后，根据 SectionDigest 生成。这样大纲才是证据驱动的，而不是凭空预设。

### Q17：draft 阶段为什么要做 seed queries？

seed queries 是最初的轻量查询。它们帮助系统快速摸清一个主题可能有哪些来源和方向。例如一个学术问题，seed queries 可能会发现综述论文、顶会论文、作者主页、PDF 和学术数据库条目。

没有 seed queries，计划就完全依赖模型先验；有了 seed queries，计划至少参考了初步来源版图。

### Q18：轻量侦察搜索和正式搜索有什么区别？

轻量侦察搜索是低成本的方向探测，目标是看有哪些来源类型和研究方向；正式搜索是按批准后的研究任务系统执行，目标是收集足够证据。

可以把轻量侦察理解成“先看地图”，正式搜索是“按路线去采集材料”。

### Q19：source landscape 是什么？

source landscape 可以理解为“来源版图”。它总结侦察阶段看到的来源类型、覆盖方向和明显缺口，比如是否有学术论文、是否有官方文档、是否有近期报告、是否存在来源不足。

ResearchExecutionPlan 会参考 source landscape，避免计划完全凭空生成。

### Q20：用户可以在审批阶段改什么？

用户可以调整：

- 研究范围；
- 时间范围；
- 来源偏好；
- 是否启用学术垂直检索；
- 要抽取的字段；
- 排除哪些来源；
- 输出语言和报告风格；
- 关注哪些评价指标。

这些反馈会进入正式执行。

---

## 4. LangGraph 工作流与 Map-Reduce

### Q21：核心工作流怎么描述？

可以这样描述：

```text
supervisor
  -> dispatch_sections
  -> section_researcher x N
  -> collect_results
  -> reflector
  -> outline_builder 或 dispatch_sections
  -> writer
```

面试时要强调：Supervisor、Researcher、Reflector、Writer 是核心角色；dispatch、collect、outline_builder 是辅助工程节点。

### Q22：Supervisor 负责什么？

Supervisor 负责把用户批准的 ResearchExecutionPlan 转换成可执行的研究任务。它会初始化状态、准备 SubTask、设置研究 track，并把任务交给 dispatch 节点并行执行。

通俗说，Supervisor 是“调度员”和“任务拆分者”。

### Q23：Researcher 负责什么？

Researcher 负责真正收集证据。它执行搜索、抓取、Document 归一化、Worker 召回和重排、LLM 抽取 KnowledgeCard，并把结果写回 ResearchState。

通俗说，Researcher 是“资料采集员”和“证据抽取员”。

### Q24：Reflector 负责什么？

Reflector 负责判断证据是否足够。它会检查每个研究任务的覆盖情况、卡片数量、来源多样性、是否存在关键缺口、是否有 degraded 任务影响主结论。如果不够，它生成补充检索任务；如果够了，就让流程进入大纲和写作。

通俗说，Reflector 是“质检员”。

### Q25：Writer 负责什么？

Writer 负责最终报告写作。它不是凭空写，而是按大纲逐节写。每节会读取对应 SectionDigest，并根据 evidence_requirements 选少量高相关原始 KnowledgeCard，用这些证据支撑正文和引用。

### Q26：OutlineBuilder 是什么角色？

OutlineBuilder 是辅助节点，负责根据 SectionDigest 生成 EvidenceOutline。它决定报告结构、每节目的、每节问题、绑定哪些证据包，以及 Writer 需要优先满足哪些证据需求。

它不是四个核心业务角色之一，但在重构后的流程里很关键，因为它让大纲由证据驱动。

### Q27：Map-Reduce 在项目里是什么意思？

Map 是把多个研究任务并行执行，Reduce 是把并行结果汇总、去重、压缩和评审。

这里的 Map-Reduce 不是 Hadoop 那种大数据框架，而是工作流设计思路。它让多个研究 track 同时跑，提高效率，也让每个 track 的证据可以单独评估。

### Q28：为什么 Map-Reduce 是按研究任务，而不是最终章节？

因为最终章节要等证据收集后生成。如果搜索前就把任务绑定成最终章节，容易把未知资料硬塞进预设结构。按研究任务并行更自然，例如“概念和定义”“代表性方法”“评价指标”“局限与趋势”可以先作为研究 track 收集证据，最后再由 OutlineBuilder 组织成报告大纲。

### Q29：LangGraph 的 reducer 解决什么问题？

多个 Researcher 并行执行时，都会往状态里写 KnowledgeCard、SubTask 状态和 section results。如果没有合并规则，后写入的结果可能覆盖先写入的结果。reducer 定义了这些字段怎么合并，比如追加、去重、按任务 ID 更新状态。

### Q30：为什么路由字段重要？

Reflector 做完判断后，会把下一步路由写入状态。这样 checkpoint 保存的不只是证据，还包括“下一步应该去哪”。如果任务中断恢复，系统可以继续执行已决定的路径，而不是重新做一次可能不稳定的判断。

---

## 5. 双层检索与 SearchService

### Q31：SearchService 是什么？

SearchService 是统一检索服务。它把普通搜索、垂直检索、抓取、归一化和初步排序组织成一条链路。下游 Researcher 不需要关心具体调用 Tavily、Exa、Serper、Bocha、DuckDuckGo fallback 还是 academic vertical，只需要拿统一的 Document。

### Q32：双层检索架构是什么？

双层检索分为：

1. GeneralSearchLayer：普通搜索层，负责广覆盖找网页和资料；
2. VerticalSearchLayer：垂直专业层，只在任务需要时触发，比如 academic。

普通层解决“找得广”，垂直层解决“找得专业”。

### Q33：为什么 academic 是 vertical plugin？

学术搜索有自己的字段和判断标准，比如作者、年份、venue、DOI、PDF 链接、引用信息。它不是普通网页搜索的简单替代品，而是专业增强层。只有用户任务涉及论文、学术数据库、顶会顶刊等需求时，academic 才需要触发。

### Q34：普通搜索层有哪些来源？

当前设计中普通层可以接 Tavily、Exa、Serper、Bocha，并保留 DuckDuckGo fallback。不同来源有不同优势：有的实时性更好，有的语义检索更好，有的覆盖面更广。多源发现可以降低单一搜索源失败或偏置的影响。

### Q35：新增搜索源怎么接入？

需要实现统一接口，把外部返回结果转换为 Document 或可归一化的中间结构。新增源不应该让下游直接依赖它的原始字段，而应该经过 DocumentNormalizer。

这就是统一协议的意义：新来源接入只影响 SearchService 的局部，不影响 Researcher、Worker、Writer。

### Q36：什么时候触发垂直检索？

可以由 ResearchExecutionPlan 的 source_strategy、SubTask 类型、用户要求、关键词和任务领域共同决定。比如用户明确说“论文”“综述”“顶会”“医学文献”“法规条文”，就应该考虑垂直层。

### Q37：多视角查询扩展是什么？

多视角查询扩展是把用户问题拆成多个搜索角度。比如“调研 AI 物理层密钥生成”不能只搜一句原问题，还要从综述、顶刊论文、会议论文、模型架构、性能指标、局限、未来趋势等角度搜。

它的目标是增加覆盖面，避免只召回某一种资料。

### Q38：查询改写和多视角查询有什么区别？

多视角查询解决“从哪些角度搜”，查询改写解决“同一个角度搜不到时怎么换一种说法”。前者偏规划，后者偏容错。

---

## 6. Document、KnowledgeCard、SectionDigest

### Q39：Document 是什么？

Document 是所有检索和抓取结果的统一结构。它至少要让下游拿到稳定的 `url`、`title`、`content`、`source_layer`、`source_kind` 等字段。

可以把 Document 理解成“清洗后的原始材料”。

### Q40：Document 为什么必须有 content 和 url？

`content` 是下游抽取证据的正文来源；`url` 是最终引用回源的基础。没有 content，Researcher 不知道从哪里抽卡；没有 url，Writer 即使写了引用，用户也不能点击回来源。

### Q41：Document 常见字段有哪些？

常见字段包括：

- `document_id`；
- `url`；
- `title`；
- `content`；
- `raw_content`；
- `source_name`；
- `source_layer`；
- `source_kind`；
- `published_time`；
- `authors`；
- `venue`；
- `year`；
- `doi`；
- `pdf_url`；
- `metadata`；
- `score`。

不是每个来源都有所有字段，但下游关键字段要稳定。

### Q42：KnowledgeCard 是什么？

KnowledgeCard 是从 Document 中抽出来的结构化证据。它不是整篇网页，而是一条可用于写作的证据卡片，通常包含：

- claim：这条证据表达的核心判断；
- summary：证据摘要；
- exact_excerpt：原文摘录；
- source：来源 URL；
- source_title：来源标题；
- confidence：置信度；
- metadata：补充元数据。

### Q43：为什么不直接把 Document 给 Writer？

Document 可能很长、很杂，有导航栏、重复段落、广告、无关上下文。直接给 Writer 会浪费上下文，也容易让模型抓错重点。KnowledgeCard 把 Document 中真正有用的论点、数据、摘录提取出来，Writer 面对的是更干净的证据。

### Q44：KnowledgeCard 的 confidence 有什么用？

confidence 表示系统对这条卡片可用性的判断。它可以影响去重、排序、证据包压缩和 Writer 选择。比如来源可靠、摘录清楚、论点明确的卡片应该优先进入写作上下文。

### Q45：SectionDigest 是什么？

SectionDigest 是按研究任务压缩后的章节证据包。它把多张 KnowledgeCard 汇总成更短、更有结构的证据包，保留主要结论、关键来源、重要摘录、冲突点和不确定性。

它的名字里有 Section，但更准确地说，它对应的是研究 track 或任务证据包，不是搜索前固定的最终报告章节。

### Q46：为什么需要 SectionDigest？

因为 KnowledgeCard 数量可能很多，不能全部塞进大纲生成和写作上下文。SectionDigest 用更大的卡片预算做广度压缩，让 OutlineBuilder 看见整体证据版图。

没有 SectionDigest，就只能在“全量卡片超上下文”和“少量卡片丢广度”之间二选一。

### Q47：SectionDigest 和 Writer 原始卡片预算有什么区别？

SectionDigest 构建时可以使用更多卡片，因为它要尽量覆盖广度。Writer 写单个章节时只选择少量高相关原始 KnowledgeCard，因为写作需要精确但不能过载。

通俗讲：SectionDigest 是给大纲看的“全局地图”，原始卡片是给正文用的“精确证据”。

### Q48：全局去重怎么理解？

不同搜索源可能返回同一篇文章，不同网页也可能重复引用同一个事实。全局去重会尽量合并重复证据，避免 Writer 看到一堆重复卡片后误以为某个观点有很多独立来源支持。

---

## 7. OutlineBuilder 与大纲

### Q49：大纲什么时候生成？

大纲在证据收集和 Reflector 评估之后生成。OutlineBuilder 会基于 SectionDigest 生成 EvidenceOutline。

这样做的原因是：只有看过真实证据后，系统才知道报告应该重点写哪些方面。

### Q50：大纲包含哪些内容？

每个大纲章节通常包含：

- `title`：章节标题；
- `purpose`：本节目的；
- `questions`：本节要回答的问题；
- `evidence_digest_ids`：本节绑定的 SectionDigest；
- `evidence_requirements`：本节需要的证据类型或取材要求。

### Q51：evidence_digest_ids 有什么作用？

`evidence_digest_ids` 把大纲章节和 SectionDigest 连接起来。Writer 写某一节时，根据这些 ID 找到相关证据包，从而避免每节都扫描所有证据。

它相当于告诉 Writer：“这一节主要从这些证据包取材。”

### Q52：evidence_requirements 有什么作用？

`evidence_requirements` 是大纲给 Writer 的取材要求。比如：

- 需要包含定量指标；
- 需要优先使用一手来源；
- 需要覆盖反方观点；
- 需要引用最新来源；
- 需要保留 exact_excerpt 支撑关键句。

它不直接指定具体卡片，而是指导 Writer 如何从原始 KnowledgeCard 里筛选。

### Q53：为什么大纲不直接指定所有卡片或链接？

因为大纲负责结构，不负责逐句证据绑定。具体写作时才知道某一段需要哪些摘录、哪些来源更合适。如果大纲阶段过早固定所有卡片，反而会限制 Writer 根据上下文选择更合适证据。

更合理的方式是：大纲绑定证据包，提出证据要求；Writer 再根据章节目标和要求选择原始卡片。

### Q54：大纲和 SectionDigest 谁更重要？

它们解决不同问题。SectionDigest 是证据的压缩表示，回答“我们搜到了什么”；大纲是报告结构，回答“应该怎么讲”。Writer 同时需要两者。

### Q55：如果 SectionDigest 覆盖不够，大纲会怎样？

OutlineBuilder 可以在大纲中体现不确定性或证据缺口，但更理想的情况是 Reflector 在大纲前发现关键缺口并触发补查。如果缺口无法补齐，报告应明确说明限制，而不是假装证据充分。

---

## 8. Writer、分章节写作与引用

### Q56：Writer 使用什么上下文？

Writer 每节使用：

1. 大纲中的章节目标和问题；
2. 该节绑定的 SectionDigest；
3. 根据 evidence_requirements 筛选出的少量高相关原始 KnowledgeCard；
4. 来源目录和引用规则。

所以 Writer 不是只看证据包，也不是看全量卡片。

### Q57：为什么要分章节写作？

分章节写作可以控制上下文规模，也能让每节聚焦自己的问题。如果整篇报告一次写，可能需要塞入大量卡片，容易超上下文，也容易结构混乱。

分章节写作还能配合 evidence_digest_ids，让每节只看相关证据包。

### Q58：Writer 为什么不能只看 SectionDigest？

SectionDigest 是压缩后的，适合概览，但可能丢失原文细节。写具体结论时，Writer 需要 exact_excerpt 和 source 这样的原始证据字段。少量原始 KnowledgeCard 可以补足细节和引用。

### Q59：Writer 为什么不能只看原始 KnowledgeCard？

原始卡片多的时候会超过上下文，而且可能让模型陷入细节，忽略整体结构。SectionDigest 提供“这一组证据整体说明什么”，大纲提供“这一节要写什么”，原始卡片提供“这句话用什么证据支撑”。

### Q60：引用最终是什么样？

正文保留可点击 [1] 小标。也就是说正文视觉上仍然是 `[1]`，但它是链接，用户点击可以打开来源。References 部分展示完整标题和 URL。

示意：

```html
正文：该方法在动态环境下提升了密钥一致性<sup><a href="https://example.com/paper">[1]</a></sup>

References：
1. [Paper Title](https://example.com/paper)
```

### Q61：为什么正文不直接显示完整标题链接？

因为正文会被长标题打断，可读性差。学术报告常见做法是正文放编号，参考文献区放完整来源。项目保留这种阅读体验，同时让编号本身可点击。

### Q62：引用校验检查什么？

引用校验主要检查：

- 正文中的引用是否有对应来源；
- 引用 URL 是否来自 source catalog；
- 是否有孤立编号；
- References 是否去重；
- 来源标题和 URL 是否可输出；
- KnowledgeCard 是否能回溯到来源。

### Q63：能不能说报告完全没有幻觉？

不要这样说。更稳的说法是：系统通过证据抽取、来源目录、可点击引用和引用校验显著降低幻觉，并让来源链接可追溯。但任何 LLM 系统都不能承诺完全没有错误。

---

## 9. Reflector、补查、饱和度与 degraded

### Q64：Reflector 为什么必要？

没有 Reflector，系统会“一轮搜索后直接写”。这很容易遗漏关键问题。Reflector 的作用是像审稿人一样检查证据是否足够，发现缺口就补查。

### Q65：Reflector 看哪些信号？

常见信号包括：

- 每个研究任务的证据数量；
- 证据覆盖的问题数；
- 来源多样性；
- 是否有一手来源或权威来源；
- 是否存在 degraded 子任务；
- 是否存在互相矛盾的证据；
- 本轮新增证据相对上一轮的增量。

### Q66：什么是跨轮次饱和度？

跨轮次饱和度用于判断继续搜索还有没有明显收益。如果新一轮搜索带来的新来源、新观点、新证据很少，说明主题可能已经接近饱和。此时继续搜只会增加成本，不一定提高报告质量。

### Q67：如果某个任务 degraded，系统会怎么处理？

degraded 表示这个任务当前没有拿到足够可用证据，但不一定导致整个研究失败。Reflector 会判断它影响大不大：

- 如果影响主结论，就生成补充检索任务；
- 如果只是边缘问题，就记录限制并继续；
- 如果多轮都无法补齐，报告里应体现不确定性。

### Q68：三层容错具体是什么？

三层容错是：

1. retry/backoff：临时错误先重试；
2. query reformulation：查询可能写得不好，就改写后再搜；
3. degraded：仍然失败就标记降级，由 Reflector 判断影响。

这样局部失败不会直接拖垮整个研究任务。

### Q69：为什么不能无限补查？

无限补查会浪费搜索额度和模型成本，也可能让任务永远不结束。系统用最大循环次数、饱和度和缺口重要性共同控制补查。只有重要缺口才值得继续查。

### Q70：如果重要缺口一直查不到怎么办？

应当在报告中明确说明这个限制，而不是编造答案。比如“现有公开来源未找到足够证据支持 X”。这比硬写一个没有来源的结论更可靠。

---

## 10. Worker、Embedding、FAISS 与 Rerank

### Q71：Worker 进程负责什么？

Worker 负责比较重的向量相关任务，包括 embedding、FAISS 召回和 CrossEncoder rerank。主进程把候选文档交给 Worker，Worker 返回排序后的高相关内容。

### Q72：为什么要独立 Worker 进程？

embedding 和 rerank 可能耗时、占内存，还可能加载较大的模型。如果放在 MCP 主进程，会阻塞工具调用。独立 Worker 可以隔离重计算，让主进程继续响应 status、result 等请求。

### Q73：FAISS 召回和 CrossEncoder rerank 的关系是什么？

FAISS 召回是先从大量文本块里快速找一批可能相关的候选；CrossEncoder rerank 再对这些候选做更精细的相关性排序。前者快，后者准，两者结合可以平衡速度和质量。

### Q74：什么是 Small-to-Big 召回？

Small-to-Big 可以理解成先用小粒度文本块找相关句子或段落，再把它们所在的更大上下文拿回来。这样既能精确匹配查询，又不丢失上下文。

### Q75：Worker 出错会不会导致整个任务失败？

通常不会。Worker 出错后，对应子任务可以标记 degraded。Reflector 再判断这个 degraded 是否影响主结论。如果影响大，就补查；如果影响小，就继续。

---

## 11. LiteLLM、多模型与角色配置

### Q76：为什么用 LiteLLM？

LiteLLM 提供统一模型调用层，可以接不同供应商。项目不需要把 OpenAI、Claude、DeepSeek 等供应商的调用逻辑写在业务代码里。

### Q77：role-based model override 是什么？

不同角色可以配置不同模型。例如：

- Planner 需要规划能力；
- Researcher 需要稳定 JSON 抽取；
- Reviewer 需要判断缺口；
- Writer 需要长文本写作。

role-based override 让这些选择通过配置完成，而不是写死在代码里。

### Q78：LiteLLM 会不会让所有模型效果一样？

不会。LiteLLM 只是统一调用接口，不会改变模型能力。不同模型在规划、抽取、长文本写作上的表现仍然不同。

---

## 12. 存储、恢复、日志与版本

### Q79：为什么用 SQLite？

当前项目是本地 MCP Server，SQLite 部署简单、无需额外服务，适合保存任务状态、事件、报告版本和 LangGraph checkpoint。如果以后做多用户服务，可以再换成 PostgreSQL。

### Q80：持久化保存了哪些东西？

主要包括：

- task registry；
- graph checkpoint；
- ResearchExecutionPlan；
- ReconnaissanceResult；
- KnowledgeCard；
- source catalog；
- SectionDigest；
- 报告文件；
- 运行日志；
- 报告版本信息。

### Q81：任务恢复怎么实现？

LangGraph checkpoint 保存图状态，task registry 保存任务元数据。任务中断后，可以根据 task_id 找回状态，从上次 checkpoint 附近继续执行，而不是从零开始。

### Q82：为什么报告和卡片要落本地文件？

报告和卡片可能很大，不适合全部塞进 MCP 响应。落盘后，MCP 工具可以返回路径、摘要和预览。用户需要完整报告时直接打开本地文件。

### Q83：日志有什么作用？

日志用于排查每个任务在哪个阶段失败、搜索源是否可用、Worker 是否超时、LLM 输出是否异常。长任务没有日志很难调试。

### Q84：版本比较怎么做？

每次正式报告可以保存成一个版本。follow-up 后生成新版本，`compare_report_versions` 可以比较两个版本的主要变化，帮助用户理解增量研究带来了什么。

---

## 13. 测试、边界与可靠性

### Q85：这次重构应该重点测什么？

重点测：

- draft 返回调研执行策略；
- SearchService 能聚合普通层和垂直层；
- 垂直层只在需要时触发；
- Document normalizer 保证 content 和 url；
- Worker 接收的 documents 都有 content；
- Writer 输出可点击 [1] 小标；
- 引用校验能发现无 URL、孤立编号和来源不存在；
- fake retriever 能跑通 draft 到 report 的 smoke test。

### Q86：为什么新增外部源第一版不做真实 API 测试？

因为新增源有 key、配额、网络、地区限制等不稳定因素。第一版重点是接口和可插拔结构，测试用 fake retriever 验证流程正确。真实 API 测试可以后续单独做集成测试。

### Q87：项目还有哪些局限？

主要局限包括：

- 搜索 API 不可用时质量会下降；
- LLM 抽取仍可能有误；
- 本地 Worker 并发能力有限；
- 引用可追溯不等于每个结论都被形式化证明；
- 垂直专业层目前以 academic 为主，更多领域插件需要继续扩展。

### Q88：如果所有搜索都失败会怎样？

系统不应该崩溃。任务会记录 runtime issue 或 degraded 状态，Reflector 判断缺口影响。最终如果无法获得足够证据，报告应明确说明来源不足，而不是编造结论。

### Q89：如何评价报告质量？

可以从过程指标和结果指标看：

- 证据覆盖度；
- 来源多样性；
- 一手来源比例；
- 关键问题是否回答；
- 引用是否可点击并回源；
- 是否记录不确定性；
- 用户是否能基于报告继续追问。

### Q90：为什么不要说“完全可追溯”？

因为系统能保证引用链接来自 source catalog，并尽量让正文引用回到证据卡片和来源，但不能保证外部网页永远可访问，也不能保证每句话都被机器形式化验证。更稳妥的说法是：证据来源链接可追溯。

---

## 14. 架构设计追问

### Q91：为什么要把大纲和写作分开？

大纲负责结构，写作负责表达。如果让 Writer 一边决定结构一边写全文，模型容易被局部材料带偏。OutlineBuilder 先根据 SectionDigest 生成结构，Writer 再按结构逐节写，职责更清楚。

### Q92：为什么 SectionDigest 构建时用更多卡片？

SectionDigest 的目的是给大纲提供广度，所以它需要看到更多卡片。它不要求每条都进入最终正文，而是尽量覆盖主题版图。

### Q93：为什么 Writer 用更少卡片？

Writer 的任务是写某一节，不需要看所有卡片。少量高相关卡片可以提高准确性，减少上下文噪音，也降低模型把无关证据混进正文的风险。

### Q94：为什么不把所有东西都放进 1M 上下文？

即使模型上下文很大，也不代表应该塞满。上下文越大，噪音越多，成本越高，模型也更难聚焦。工程上更好的方式是先压缩、再按章节选择相关证据。

### Q95：如果面试官问“你这个大纲是不是还是模型生成的”，怎么答？

可以答：是模型生成，但不是凭空生成。它的输入不是用户一句话，而是 SectionDigest，也就是证据收集后的压缩证据包。模型负责组织结构，证据包负责约束内容。

### Q96：如果问“引用编号由谁决定”，怎么答？

Writer 或引用处理逻辑会根据选用来源建立编号映射。正文里的 `[1]` 对应 source catalog 中的一个 URL，References 再展示完整标题链接。关键不是编号本身，而是编号能映射到真实来源。

### Q97：如果问“正文点击 [1] 是前端做还是后端做”，怎么答？

当前可以在 Markdown 输出中直接生成 HTML 上标链接，也可以由前端渲染时把引用编号转成链接。项目核心要求是：正文显示 [1]，用户点击 [1] 能到来源，References 显示完整来源。

### Q98：如果问“为什么不是所有引用都放 References，正文不加链接”，怎么答？

正文可点击能减少用户查找成本。用户看到一个关键结论时，可以立刻点旁边的 [1] 回源；References 则用于完整整理。

### Q99：如果问“source catalog 从哪里来”，怎么答？

source catalog 来自 Document 和 KnowledgeCard 的来源信息。每个可引用来源都会记录 URL、标题、来源类型等。Writer 生成引用时只能引用 catalog 中存在的来源。

### Q100：如果问“证据卡片太多怎么办”，怎么答？

先全局去重，再按任务压缩成 SectionDigest。大纲看 SectionDigest；写作按章节筛选少量原始 KnowledgeCard。这样既保留广度，又控制上下文。

---

## 15. 扩展性与未来改进

### Q101：如果要增加医学文献垂直检索，怎么做？

新增一个 medical vertical plugin，封装 PubMed 或其他医学数据库，返回统一 Document。触发逻辑可以根据用户问题、source_strategy 或任务类型判断。下游不需要改，因为它仍然消费 Document。

### Q102：如果要增加专利或法规检索，怎么做？

和医学类似，新增对应 vertical plugin。关键是把专业字段放入 metadata，同时保证 url、title、content 等基础字段稳定。

### Q103：如果要做多租户 SaaS，需要改什么？

需要把 SQLite 换成更适合并发的数据库，例如 PostgreSQL；增加用户和租户隔离；Worker 改成队列和 Worker pool；MCP 或外层 API 加认证和限流。

### Q104：如果要提升报告质量，优先改哪里？

优先改三处：

1. 更强的证据抽取和校验；
2. 更好的 SectionDigest 压缩策略；
3. Writer 对 evidence_requirements 的利用。

这三处直接影响报告是否有证据、结构是否合理、引用是否准确。

### Q105：如果要降低成本，优先改哪里？

可以优化：

- 侦察搜索和正式搜索配额；
- Worker 召回数量；
- SectionDigest 卡片预算；
- Writer 每节原始卡片数量；
- role-based model override，用便宜模型处理简单阶段。

### Q106：如果要提升速度，优先改哪里？

可以优化并行度、搜索超时、Worker 批处理、缓存和早停策略。也可以对低优先级任务减少搜索预算，让核心任务优先完成。

### Q107：如果要提升可解释性，优先改哪里？

可以增强运行日志、保存每轮 Reflector 决策、保存每节 Writer 使用的 evidence_digest_ids 和原始卡片列表，并在报告附录输出证据映射。

### Q108：如果要做更强的引用校验，怎么做？

可以增加：

- 链接可访问性检查；
- exact_excerpt 和正文句子的语义一致性检查；
- 引用覆盖率统计；
- 来源快照保存；
- 每节引用密度规则。

---

## 16. 常见高危回答修正

### Q109：高危说法：系统先生成报告大纲，再开始搜索。

修正：系统先生成调研执行策略，用户确认后正式搜索。大纲是在证据收集之后，由 OutlineBuilder 基于 SectionDigest 生成。

### Q110：高危说法：Writer 只看章节证据包。

修正：Writer 按大纲写，每节读取对应 SectionDigest，并筛选少量高相关原始 KnowledgeCard。SectionDigest 提供广度，原始卡片提供精确证据。

### Q111：高危说法：正文引用是完整标题链接。

修正：正文是可点击 [1] 小标，References 展示完整标题链接。

### Q112：高危说法：学术搜索和普通搜索引擎放在同一层。

修正：普通搜索层负责广覆盖，academic 属于垂直专业层，只在需要学术资料时触发。

### Q113：高危说法：degraded 后就不再处理。

修正：degraded 后交给 Reflector 判断影响。重要缺口会触发补查；不重要的缺口会记录限制。

### Q114：高危说法：有引用就不会幻觉。

修正：引用能显著降低幻觉并提供来源回溯，但仍要靠证据抽取、来源目录、引用校验和人工阅读共同保证质量。

### Q115：高危说法：四个核心角色就是代码里的全部节点。

修正：四个核心角色是业务职责，真实 LangGraph 还包含 dispatch、collect、outline_builder 等辅助节点。

---

## 17. 面试结尾总结

### Q116：如果最后让你总结项目亮点，怎么说？

可以这样回答：

> 这个项目最重要的不是调了几个搜索 API，而是把复杂调研变成了证据驱动的工程流程。它通过 MCP 暴露标准工具，先让用户确认调研执行策略；内部用 LangGraph 编排 Supervisor、Researcher、Reflector、Writer；检索层用 SearchService 区分普通搜索和垂直专业检索；证据层从 Document 抽取 KnowledgeCard，再压缩成 SectionDigest；大纲基于证据生成，Writer 按大纲和证据包分章节写作；最后用可点击 [1] 小标和 References 链接保证来源可回溯。再加上 SQLite checkpoint、Worker 异步召回重排和三层容错，它更像一个可靠的研究系统，而不是简单总结脚本。

### Q117：如果面试官只给 30 秒，你怎么讲？

可以这样说：

> 我做的是一个本地 Deep Research MCP Server。它先根据用户问题生成调研执行策略并让用户确认，然后用双层检索收集资料，把资料统一成 Document，抽取 KnowledgeCard，压缩成 SectionDigest，再由 LangGraph 的 Reflector 判断是否补查。证据足够后，系统基于 SectionDigest 生成大纲，Writer 按大纲和证据分章节写报告，并输出可点击 [1] 小标和完整 References。

### Q118：如果问“你最满意的设计是什么”，怎么答？

推荐答 SectionDigest 和证据驱动大纲：

> 我最满意的是把全量卡片和最终写作之间加了一层 SectionDigest。它解决了上下文容量和证据广度之间的矛盾：大纲生成需要广度，所以看更大的压缩证据包；章节写作需要精度，所以再取少量高相关原始卡片。这样比直接把所有卡片扔给模型更稳。

### Q119：如果问“最大的难点是什么”，怎么答？

可以答流程一致性和证据上下文控制：

> 难点不是单次搜索，而是长流程里每一步的产物要能接上下一步。搜索结果要统一成 Document，抽卡要保留来源，SectionDigest 要压缩但不能丢关键证据，大纲要绑定证据包，Writer 要能回到原始卡片做引用。任何一个字段设计不好，后面都会接不上。

### Q120：如果问“你从这个项目学到了什么”，怎么答？

可以这样答：

> 我最大的收获是，复杂 LLM 应用不能只靠 prompt，要把模型放进可控的工程流程里。哪些步骤让模型做，哪些步骤用结构化数据约束，哪些结果要持久化，哪些失败要降级，这些工程设计决定了系统能不能从 demo 变成可用工具。
