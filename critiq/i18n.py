import os
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    ENGLISH = "en"
    CHINESE = "zh"


@dataclass
class I18nPrompt:
    WARMUP_ZERO_ONE_PROMPT_TEMPLATE_0: str
    WARMUP_ZERO_ONE_PROMPT_TEMPLATE_1: str
    WARMUP_PAIR_PROMPT_TEMPLATE_AB: str
    WARMUP_PAIR_PROMPT_TEMPLATE_BA: str
    MANAGER_PROMPT_POSTFIX: str
    MANAGER_PROMPT_TEMPLATE: str
    ACCURACY_PROMPT: str
    GOOD_CRITERIA_PROMPT_TEMPLATE: str
    MID_CRITERIA_PROMPT_TEMPLATE: str
    MID_CRITIQUE_PROMPT: str
    MID_A_PROMPT_TEMPLATE: str
    MID_B_PROMPT_TEMPLATE: str
    MID_01_PROMPT_TEMPLATE: str
    MID_HOWEVER_PROMPT_TEMPLATE: str
    MID_REFLECTION_PROMPT: str
    MID_0_FOR_1_PROMPT: str
    MID_1_FOR_0_PROMPT: str
    MID_REFINE_PROMPT_TEMPLATE: str
    MID_FORMAT_PROMPT_TEMPLATE: str
    LOW_PROMPT_TEMPLATE: str
    LOW_FORMAT_PROMPT: str
    BASELINE_WORKER_PROMPT_POSTFIX: str
    BASELINE_WORKER_PROMPT: str
    PAIR_WORKER_PROMPT_POSTFIX: str
    PAIR_WORKER_PROMPT: str
    ZERO_ONE_WORKER_PROMPT_POSTFIX: str
    ZERO_ONE_WORKER_PROMPT: str
    CRITERION_NAME_DESC_FORMAT_TEMPLAT: str


ENGLISH_PROMPT = I18nPrompt(
    WARMUP_ZERO_ONE_PROMPT_TEMPLATE_0="[DATA]\n{text}\n[/DATA]\n\nWhy is the data piece of low quality?",
    WARMUP_ZERO_ONE_PROMPT_TEMPLATE_1="[DATA]\n{text}\n[/DATA]\n\nWhy is the data piece of high quality?",
    WARMUP_PAIR_PROMPT_TEMPLATE_AB="There are two data pieces.\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]\n\nWhy is A better than B?",
    WARMUP_PAIR_PROMPT_TEMPLATE_BA="There are two data pieces.\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]\n\nWhy is B better than A?",
    MANAGER_PROMPT_POSTFIX="""\n\nYour response should be in the following **json** format:
```json
{
    "name_of_the_criterion": "Detailed description for the criterion such as what it is, how it can be evaluated, when it is applicable, or other relevant information... Be specific and detailed while keep concise.",
    ...
}
```""",
    MANAGER_PROMPT_TEMPLATE="Give {n_criteria} criteria for evaluating data quality.",
    ACCURACY_PROMPT="The worker agents had evaluated data pairs aginst these criteria. The accuracy of each criterion is as follows:",
    GOOD_CRITERIA_PROMPT_TEMPLATE="Accuracies of criteria {criteria} are over {threshold}. They are good criteria.",
    MID_CRITERIA_PROMPT_TEMPLATE="The accuracy of {criterion_name} is over {threshold_0} but less than {threshold_1}. It can be improved. Here is the raw description of the criterion:",
    MID_CRITIQUE_PROMPT="This is an incorrect case:",
    MID_A_PROMPT_TEMPLATE="[BEGIN_OF_A]\n{}\n[/END_OF_A]",
    MID_B_PROMPT_TEMPLATE="[BEGIN_OF_B]\n{}\n[/END_OF_B]",
    MID_01_PROMPT_TEMPLATE="[DATA]\n{text}\n[/DATA]",
    MID_HOWEVER_PROMPT_TEMPLATE="Against this criterion, the worker agent chose {wrong} as better, but the correct answer is {correct}. Here is how the worker agent thinks:\n\n{thought}",
    MID_REFLECTION_PROMPT="""Please analyze this incorrect case together with the worker agent's thought. Based on your anaylsis, please provide your critique for how to write a better description of this critierion to guide the worker make correct judgment or properly indicate inapplicable situations for this criterion.

Your response should be in the following **json** format:
{
    "analysis": "Your analysis here.",
    "critique": "How this criterion can be improved. Please just point out the key points in a few sentences."
}""",
    MID_0_FOR_1_PROMPT="This above data piece is mistaken as of low quality, but actually it is of high quality.",
    MID_1_FOR_0_PROMPT="This above data piece is mistaken as of high quality, but actually it is of low quality.",
    MID_REFINE_PROMPT_TEMPLATE="There are the critiques for the wrong choices.\n\n{}\n\nBased on the above critiques, please improve the description for this criterion to make worker agents get higher accuracy. For exmaple, what it is, how it can be evaluated, when it is applicable, and other relevant information. Be specific and detailed while keep concise.",
    MID_FORMAT_PROMPT_TEMPLATE='Return the improved description in the following **json** format:\n\n{{"{criterion_name}": "The improved description"}}',
    LOW_PROMPT_TEMPLATE="Criteria {criteria} have an accuracy of less than {threshold_0}. They should be removed from the criteria list. Please provide {num} new criteria. The new ones should not be duplicated with the above ones.",
    LOW_FORMAT_PROMPT="""Return the new criteria in the following **json** format:\n\n
```json
{
    "your_better_criterion_here": "Detailed description for the criterion, including what it is, how it can be evaluated, when it is applicable, and other relevant information. Be specific and detailed while keep concise.",
    ...
}
```""",
    BASELINE_WORKER_PROMPT_POSTFIX="""

Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A based on the given criteria.",
    "analysis_b": "Analyze B based on the given criteria.",
    "thought": "Compare A and B.",
    "answer": "A / B"
}
```
""",
    BASELINE_WORKER_PROMPT="Analyze the following two texts based on the given criteria:\n\n{C}\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]",
    PAIR_WORKER_PROMPT_POSTFIX="""
Your response should be in the following **JSON** format:
```json
{
    "analysis_a": "Analyze A based on the given criterion.",
    "analysis_b": "Analyze B based on the given criterion.",
    "thought": "Compare A and B.",
    "answer": "A / B / None"
}
```
Return None if any of the following conditions are met:
- The criterion is not applicable to this pair of data pieces.
- They are of the same quality.
- You are unsure.
""",
    PAIR_WORKER_PROMPT="Which is better in the aspect of **{criterion}**?\n\n{criterion}: {description}\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]",
    ZERO_ONE_WORKER_PROMPT_POSTFIX="""

Your response should be in the following **JSON** format:
```json
{
    "thought": "Your analysis.",
    "answer": "Y / N / UNSURE"
}
```
"Y" is for **yes** and "N" is for **no**. If you are unsure, answer **UNSURE**.""",
    ZERO_ONE_WORKER_PROMPT="Is this data piece of high quality in the aspect of **{criterion}**?\n\n{criterion}: {description}\n\n[DATA]\n{text}\n[/DATA]",
    CRITERION_NAME_DESC_FORMAT_TEMPLAT="\n\n[CRITERION]\nCriterion: {name}\n\nDescription: {desc}\n[/CRITERION]",
)


CHINSES_PROMPT = I18nPrompt(
    WARMUP_ZERO_ONE_PROMPT_TEMPLATE_0="[DATA]\n{text}\n[/DATA]\n\n为什么这个数据片段质量较低？",
    WARMUP_ZERO_ONE_PROMPT_TEMPLATE_1="[DATA]\n{text}\n[/DATA]\n\n为什么这个数据片段质量较高？",
    WARMUP_PAIR_PROMPT_TEMPLATE_AB="以下两个数据片段：\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]\n\n为什么 A 比 B 质量更高？",
    WARMUP_PAIR_PROMPT_TEMPLATE_BA="以下两个数据片段：\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]\n\n为什么 B 比 A 质量更高？",
    MANAGER_PROMPT_POSTFIX="""\n\n请按照如下 JSON 格式回复：
```json
{
    "示例指标": "指标的详细描述，如指标的定义以及如何评估。",
    ...
}
```""",
    MANAGER_PROMPT_TEMPLATE="给出 {n_criteria} 条用于判断数据质量的指标。",
    ACCURACY_PROMPT="各项指标的正确率如下：",
    GOOD_CRITERIA_PROMPT_TEMPLATE="这些指标的正确率超过了 {threshold}：{criteria}。它们是好的指标。",
    MID_CRITERIA_PROMPT_TEMPLATE="指标“{criterion_name}”的正确率在 {threshold_0} 和 {threshold_1}之间。以下是依据这些指标判断错误的案例：",
    MID_CRITIQUE_PROMPT="以下是一个错误案例：",
    MID_A_PROMPT_TEMPLATE="[BEGIN_OF_A]\n{}\n[/END_OF_A]",
    MID_B_PROMPT_TEMPLATE="[BEGIN_OF_B]\n{}\n[/END_OF_B]",
    MID_01_PROMPT_TEMPLATE="[DATA]\n{text}\n[/DATA]",
    MID_HOWEVER_PROMPT_TEMPLATE="依据这一指标，worker agent 认为 {wrong} 更好。然而，正确答案是 {correct}。以下是 worker agent 的分析：\n\n{thought}",
    MID_REFLECTION_PROMPT="""请分析这个错误案例以及 worker agent 的分析。基于你的分析，请提供如何改进该指标描述的批判性意见，以指导 worker agent 避免错误或精确识别指标不适用的情况。

请按照如下 JSON 格式回复：
{
    "analysis": "你的分析。",
    "critique": "该指标如何改进。请简明扼要地指出关键点。"
}""",
    MID_0_FOR_1_PROMPT="这是高质量数据，但依据该指标，被错误地判断为不符合标准。",
    MID_1_FOR_0_PROMPT="这是低质量数据，但依据该指标，被错误地判断为符合标准。",
    MID_REFINE_PROMPT_TEMPLATE="对于错题的评价如下：\n\n{}\n\n基于以上评价，请改进该指标的描述，以提高正确率。",
    MID_FORMAT_PROMPT_TEMPLATE='请按照如下 JSON 格式列出改进后的描述：\n\n{{"{criterion_name}": "改进后的描述。"}}',
    LOW_PROMPT_TEMPLATE="以下指标的正确率低于 {threshold_0}：{criteria}。他们应当从指标列表中去掉。请提供 {num} 个新指标。新指标不应与以上所有指标重复。",
    LOW_FORMAT_PROMPT="""按照如下 JSON 格式返回新的指标：\n\n
```json
{
    "更好的指标": "新指标的描述。",
    ...
}
```""",
    BASELINE_WORKER_PROMPT_POSTFIX="""

请按照如下 JSON 格式回复：
```json
{
    "analysis_a": "基于给定指标分析 A。",
    "analysis_b": "基于给定指标分析 B。",
    "thought": "比较 A 和 B。",
    "answer": "A / B"
}
```""",
    BASELINE_WORKER_PROMPT="基于给定指标比较数据 A 和 B。\n\n{C}\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]",
    PAIR_WORKER_PROMPT_POSTFIX="""

请按照如下 JSON 格式回复：
```json
{
    "analysis_a": "基于给定指标分析 A。",
    "analysis_b": "基于给定指标分析 B。",
    "thought": "比较 A 和 B。",
    "answer": "A / B / None"
}

如果存在以下情况，请回复 None：
- 该指标不适用于判断这对数据
- A 和 B 的质量相同
- 你不确定
```""",
    PAIR_WORKER_PROMPT="考虑指标“**{criterion}**”，哪段数据的质量更高？\n\n{criterion}：{description}\n\n[DATA_A]\n{A}\n[/DATA_A]\n\n[DATA_B]\n{B}\n[/DATA_B]",
    ZERO_ONE_WORKER_PROMPT_POSTFIX="""

请按照如下 JSON 格式回复：
```json
{
    "thought": "你的分析。",
    "answer": "Y / N / UNSURE"
}
```
“Y” 表示高质量，“N” 如果你不确定，请在 answer 里回复 **UNSURE**.""",
    ZERO_ONE_WORKER_PROMPT="基于指标“**{criterion}**”，这段数据是高质量数据吗？\n\n{criterion}：{description}\n\n[DATA]\n{text}\n[/DATA]",
    CRITERION_NAME_DESC_FORMAT_TEMPLAT="\n\n[CRITERION]\n指标：{name}\n\n描述：{desc}\n[/CRITERION】",
)

WORKFLOW_LANGUAGE = Language(os.getenv("WORKFLOW_LANG", "en"))

print(
    f"Setting workflow prompt language to `{WORKFLOW_LANGUAGE}`. You can change this by setting the WORKFLOW_LANG environment variable."
)

i18n_prompts: dict[Language, I18nPrompt] = {
    Language.ENGLISH: ENGLISH_PROMPT,
    Language.CHINESE: CHINSES_PROMPT,
}

local_prompts = i18n_prompts[WORKFLOW_LANGUAGE]
