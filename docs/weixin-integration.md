# Weixin ClawBot 私聊接入

## 功能范围

本项目通过微信 ilink Bot 长轮询 API 接收普通微信私聊，并仅支持以下本地只读命令：

- `帮助`：显示命令列表。
- `状态`：显示微信服务与日报配置状态，不显示任何凭据。
- `日报`：读取已有 `output/history.json`，返回最新日报元数据摘要。
- `完整报告`：将最新的 `summary_*.html` 报告加密上传并作为文件回复。

微信命令不会重新抓取 X、不会调用 DeepSeek、不会修改配置，也不能指定任意本地文件路径。本阶段不支持微信群、主动推送或定时任务。

## 协议来源与许可证

协议和媒体上传流程参考 [liiiiwh/weixin-clawbot-skill](https://github.com/liiiiwh/weixin-clawbot-skill)，参考项目采用 MIT License，Copyright (c) 2025。Python 实现按目标项目架构重新编写，并保留本说明作为来源与许可证归属记录。

使用的官方服务地址固定为：

- `https://ilinkai.weixin.qq.com`
- `https://novac2c.cdn.weixin.qq.com/c2c`

扫码响应或微信消息不能覆盖这些可信主机。

## 配置

在本机 `config.json` 中启用非敏感开关：

```json
{
  "weixin": {
    "enabled": true
  }
}
```

Bot Token 不写入 `.env` 或 `config.json`。扫码后凭据只保存在私人目录：

```text
data/weixin/
├── credentials.json
├── sync_state.json
├── users.json
├── processed_messages.jsonl
├── send_history.jsonl
└── temporary/
```

相对目录以项目根目录解析。`data/` 已被 Git 忽略；凭据和状态文件在 macOS 上使用 `0600` 权限。

## 二维码登录

macOS 示例：

```bash
cd /Users/yi/Documents/II_PROJECT/FORME/x_ailist_summerize
.venv/bin/python -m app.weixin_auth
```

命令会在 `data/weixin/temporary/` 创建本地二维码 PNG，并在终端打印本地路径。使用普通微信扫码并确认。成功后终端只显示：

```text
Weixin credentials configured: true
```

登录成功或失败后二维码文件都会被删除。扫码最长等待约 8 分钟，可按 `Ctrl+C` 安全取消。Token 过期时重新执行同一命令即可覆盖旧的失效凭据。

## 启动和停止

```bash
cd /Users/yi/Documents/II_PROJECT/FORME/x_ailist_summerize
.venv/bin/python -m app.weixin_service
```

服务使用 `get_updates_buf` 从上次位置继续长轮询，稳定消息 ID 用于去重。每条消息处理成功后才记录完成；整批消息都处理成功后才提交新游标。单条失败不会终止进程，也不会错误推进游标。

使用 `Ctrl+C` 停止。当前不提供 launchd、进程守护、macOS 自动启动或每天上午 9 点定时推送。

## context_token 与文件安全

每次回复优先使用当前入站消息携带的 `context_token`。Token 按微信用户隔离保存；缺少 Token 时不会发送，也不会借用其他用户的 Token。

“完整报告”只能发送项目 `output/` 下真实存在且非符号链接的 `.html` 文件。实现会阻止路径穿越、绝对路径注入、符号链接逃逸以及发送配置、Cookie、日志、源码或 `data/` 文件。临时加密文件在上传成功或失败后都会删除。

HTML 文件作为微信文件发送，可以下载；是否能在手机端直接预览取决于微信客户端能力。

