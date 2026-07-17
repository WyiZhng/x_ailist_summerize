# 中文内容生成规则

Prompt 版本为 `content_filter_zh_v1`、`event_candidate_zh_v1`、`pair_cluster_zh_v1` 和 `event_synthesis_zh_v1`。用户可见字段必须为简体中文，OpenAI、Claude、Gemini、DeepSeek 等官方名称保留英文；数字、日期、模型版本和 URL 不翻译。

结构化输出只接受 JSON（可受控去除代码围栏），并严格检查 `keep`、英文枚举、中文标题/摘要/理由。中文验证器拒绝全英文、密钥、Token、绝对路径和夸张营销词。DeepSeek 不可用或结构校验失败时，管线保留原始来源并使用确定性中文降级结果，不会让抓取、调度或微信重试触发额外 LLM 调用。
