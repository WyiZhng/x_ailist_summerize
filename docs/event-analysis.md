# 中文事件分析

`python -m app.event_pipeline --date YYYY-MM-DD --max-items 20` 只读取已保存的帖子和文章，不会抓取 X 或网页。管线先构造带 `<UNTRUSTED_SOURCE>` 边界的输入，再规则过滤、DeepSeek 结构化语义过滤、精确来源聚合、稳定 ID 排序和 Top 5 标记。`--no-llm` 保持完全离线，`--mock` 使用确定性 Mock。事件数据和 LLM 缓存保存在 Git 忽略的 `data/events/`。

缓存键包含输入哈希、Prompt 版本、Schema 版本和模型名称。缓存命中仍会重新检查 JSON 与中文字段；仅记录真实 API 返回的 prompt/completion Token，不自行估算。达到请求或 Token 预算后保留已完成结果并停止新增调用。

评分上限为 100，互动量仅作为很小的辅助信号；同一文章或相同文本不会重复计分。每个事件保留帖子、文章和 URL 引用。相同输入按稳定哈希产生相同事件 ID。
