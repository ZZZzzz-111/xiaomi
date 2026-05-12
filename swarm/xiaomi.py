#!/usr/bin/env python3
"""
产品洞察蜂群 (Product Insight Swarm)
多智能体用户反馈分析系统 - 完整可运行版
依赖: pip install crewai langchain-openai
运行前设置环境变量: export OPENAI_API_KEY="sk-xxx"
"""

import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from textwrap import dedent

# ============================================================
# 0. 初始化大模型（请确保已设置 OPENAI_API_KEY）
# ============================================================
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# ============================================================
# 1. 定义专业 Agent (6个)
# ============================================================

# 1.1 数据清洗 Agent
cleaner = Agent(
    role="反馈清洗专家",
    goal="对原始用户反馈进行去噪、标准化和分类，提取结构化信息。",
    backstory=dedent("""\
        你是一名经验丰富的运维分析师，擅长从杂乱的多渠道文本中识别有效信息，
        过滤掉无意义的噪音和重复内容，并按照产品功能模块进行归类。"""),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# 1.2 情绪分析 Agent
sentiment_agent = Agent(
    role="情绪分析专家",
    goal="识别每条反馈背后的用户情绪（愤怒/失望/欣喜/期待），并给出情绪强度评分(1-5)。",
    backstory=dedent("""\
        你拥有心理学背景，专精于文本情感计算，能准确捕捉用户字里行间的
        隐含情绪，为产品决策提供情感维度参考。"""),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# 1.3 主题聚合 Agent
topic_agent = Agent(
    role="主题聚合专家",
    goal="从大量反馈中提取高频话题和共性问题，形成主题聚类。",
    backstory=dedent("""\
        你擅长使用自然语言处理技术进行话题建模，能够识别出海量
        反馈中反复出现的关键词和模式，归纳为可行动的问题主题。"""),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# 1.4 因果推理 Agent (长链推理核心)
causal_agent = Agent(
    role="因果推理专家",
    goal="对关键问题执行'现象-根因-业务影响'的长链推理，找出深层原因。",
    backstory=dedent("""\
        你是一名资深系统架构师和业务分析师，习惯用因果链追溯问题本质。
        你总能通过多层追问，找到问题背后的技术债或流程缺陷，
        并评估其对用户留存和商业指标的具体影响。"""),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# 1.5 优先级裁判 Agent
priority_agent = Agent(
    role="需求优先级裁判",
    goal="根据业务价值、紧急度、实现成本等维度，对所有洞察排序，输出P0-P3等级。",
    backstory=dedent("""\
        你是一名拥有多年经验的产品总监，深谙商业策略和用户价值平衡。
        你能够犀利地判断哪些需求应该立即行动，哪些可以放入待办清单。"""),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# 1.6 叙事编排 Agent (报告生成)
reporter = Agent(
    role="产品叙事官",
    goal="将所有分析结果整合为一份可读性强的周报，包含数据洞察和执行建议。",
    backstory=dedent("""\
        你是一名顶级的产品营销经理，擅长用故事化的方式呈现数据，
        让团队成员一眼看懂本周最重要的三件事和下一步动作。"""),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ============================================================
# 2. 模拟原始反馈数据 (可按需替换)
# ============================================================
raw_feedback = """
1. 更新后App打开白屏，闪退了5次，根本用不了！一星差评！
2. 希望增加夜间模式，晚上刷手机眼睛疼。
3. 支付时一直转圈圈，钱扣了订单却没生成，客服找不到人，愤怒！
4. 新上线的推荐算法太准了，我连续买了三单，好评！
5. 搜索功能经常搜不到已经存在的商品，比如搜“苹果”出来一堆手机壳。
6. 退货流程太复杂了，填了三次单子还没通过，要失去耐心了。
7. 为什么竞品X都能微信一键登录，你们还要输手机号？赶紧优化！
8. 首页推荐的内容总是不感兴趣，点了不感兴趣还继续推，感觉像个瞎子。
9. 整体体验还行，就是有时候消息通知延迟很久。
10. 活动页面的优惠券点进去显示已过期，太坑了！
"""

# ============================================================
# 3. 定义任务链 (顺序执行，部分依赖前序输出)
# ============================================================

# 阶段1: 清洗
task_clean = Task(
    description=f"清洗以下原始反馈，去除无意义内容，并按功能模块分类输出结构化列表：\n{raw_feedback}",
    expected_output="一个结构化JSON列表，字段包含：id, 原始文本摘要, 功能模块, 是否有效。",
    agent=cleaner,
)

# 阶段2: 情绪分析 (依赖清洗后的有效反馈)
task_sentiment = Task(
    description="基于上一阶段清洗后的有效反馈，进行情绪分析，为每条标注情绪类别和强度。",
    expected_output="每个反馈ID对应的情绪标签(愤怒/失望/欣喜/期待)和强度(1-5)。",
    agent=sentiment_agent,
)

# 阶段3: 主题聚合
task_topic = Task(
    description="基于有效反馈，提取3-5个高频主题，并列出每个主题下的典型反馈ID。",
    expected_output="主题列表，每个主题包含：主题名称、关键词、涉及反馈ID、严重程度。",
    agent=topic_agent,
)

# 阶段4: 因果推理 (长链推理)
task_causal = Task(
    description=dedent("""\
        选择排名前2的关键主题（如支付失败、搜索不准），进行深入推理分析。
        对每个主题，请严格按'现象→直接原因→根本原因→业务影响'的链条输出。
        至少追溯两层原因，并评估影响（如：支付失败 → 资金安全恐慌 → 用户流失）。"""),
    expected_output="两个长链推理段落，每个包含完整的因果链条和量化影响评估。",
    agent=causal_agent,
)

# 阶段5: 优先级排序
task_priority = Task(
    description="结合情绪分析、主题聚合和因果推理的结果，对所有问题打出P0/P1/P2/P3优先级。",
    expected_output="优先级列表，每个条目包含：问题主题、优先级、建议处理SLA、理由。",
    agent=priority_agent,
)

# 阶段6: 生成最终周报
task_report = Task(
    description=dedent("""\
        综合所有分析成果，生成一份产品洞察周报，要求包含：
        1. 本周核心发现（一段话总结）
        2. 情绪分布概况
        3. TOP3 关键问题及因果摘要
        4. 优先级行动清单（表格形式）
        5. 本周正能量（用户点赞的功能）
        请用专业的Markdown格式输出，方便直接群发。"""),
    expected_output="一份完整的Markdown格式周报。",
    agent=reporter,
)

# ============================================================
# 4. 组装 Crew 并运行
# ============================================================
crew = Crew(
    agents=[cleaner, sentiment_agent, topic_agent, causal_agent, priority_agent, reporter],
    tasks=[task_clean, task_sentiment, task_topic, task_causal, task_priority, task_report],
    process=Process.sequential,  # 严格按顺序执行，后置任务可看到前置结果
    verbose=True,
)

if __name__ == "__main__":
    print("🚀 产品洞察蜂群启动...")
    result = crew.kickoff()
    print("\n" + "="*60)
    print("📊 最终生成的产品洞察周报如下：")
    print("="*60)
    print(result)