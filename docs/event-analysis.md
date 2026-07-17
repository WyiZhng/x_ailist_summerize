# 中文事件分析

`python -m app.event_pipeline --date YYYY-MM-DD --max-items 20 --no-llm` 只读取已保存的帖子和文章，不会抓取 X 或网页。管线先构造带 `<UNTRUSTED_SOURCE>` 边界的输入，再规则过滤、精确来源聚合、稳定 ID 排序和 Top 5 标记。事件数据保存在 Git 忽略的 `data/events/`。

评分上限为 100，互动量仅作为很小的辅助信号；同一文章或相同文本不会重复计分。每个事件保留帖子、文章和 URL 引用。相同输入按稳定哈希产生相同事件 ID。
