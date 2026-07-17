# 中文内容生成规则

Prompt 版本为 `content_filter_zh_v1`、`event_candidate_zh_v1`、`pair_cluster_zh_v1` 和 `event_synthesis_zh_v1`。用户可见字段必须为简体中文，OpenAI、Claude、Gemini、DeepSeek 等官方名称保留英文；数字、日期、模型版本和 URL 不翻译。

结构化输出只接受 JSON（可受控去除代码围栏），并严格检查 `keep`、英文枚举、中文标题/摘要/理由。中文验证器拒绝全英文、密钥、Token、绝对路径和夸张营销词。正常 `keep=false` 记为过滤；超时和限流可恢复重试；超过重试上限的结构或中文验证错误记为永久无效。失败项不会被伪装为中文事件，成功事件仍可生成部分成功报告。
