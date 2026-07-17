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

X List、LLM 和 X Cookie 可以写入项目根目录已忽略的 `.env`，例如 `XLS_X_LIST_URLS`、`XLS_LLM_PROVIDER`、`XLS_LLM_MODEL`、`XLS_LLM_API_KEY`、`XLS_X_AUTH_TOKEN` 与 `XLS_X_CT0`。程序只读取未设置的 `XLS_*` 值，不执行 `.env` 内容；launchd 也通过同一机制读取，凭据不会进入 plist。

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

使用 `Ctrl+C` 停止。也可使用下文 launchd 管理自动启动。

## context_token 与文件安全

每次回复优先使用当前入站消息携带的 `context_token`。Token 按微信用户隔离保存；缺少 Token 时不会发送，也不会借用其他用户的 Token。

“完整报告”只能发送项目 `output/` 下真实存在且非符号链接的 `.html` 文件。实现会阻止路径穿越、绝对路径注入、符号链接逃逸以及发送配置、Cookie、日志、源码或 `data/` 文件。临时加密文件在上传成功或失败后都会删除。

HTML 文件作为微信文件发送，可以下载；是否能在手机端直接预览取决于微信客户端能力。

## 订阅、推送和调度

私聊发送 `订阅每日推送`（也支持 `订阅日报`、`开启每日推送`）开启；`取消每日推送`、`订阅状态` 和 `补发日报` 分别用于关闭、查询和补发。计划使用 `Asia/Shanghai` 的每日 09:00，补跑窗口为 09:00–12:00。Mac 关机时无法运行；睡眠唤醒后只在窗口内补跑。

```bash
.venv/bin/python -m app.weixin_push --status
.venv/bin/python -m app.weixin_push --preview-latest --date YYYY-MM-DD
.venv/bin/python -m app.weixin_push --send-test
.venv/bin/python -m app.weixin_push --send-latest --date YYYY-MM-DD
.venv/bin/python -m app.daily_delivery --run-now
.venv/bin/python -m app.daily_delivery --retry-failed
.venv/bin/python -m app.daily_delivery --status
```

预览和发送都只读取已保存的结构化事件与 HTML，不调用 DeepSeek。预览仅显示通过质量闸门的中文 Top 5，不输出用户 ID、Token 或本地路径。报告生成与微信通知状态独立：通知失败只补发，不再次抓取 X 或调用 DeepSeek。主动推送使用每位用户最近私聊的 `context_token`，失效后请发送任意消息刷新；其长期有效期尚未验证。`--send-latest --force` 仅供本机管理员明确重发，并会打印警告。

## macOS 自动运行

```bash
.venv/bin/python -m app.launchd_manager install
.venv/bin/python -m app.launchd_manager status
.venv/bin/python -m app.launchd_manager restart
.venv/bin/python -m app.launchd_manager uninstall
```

工具只管理 `~/Library/LaunchAgents` 中本项目的两个 plist，不写入 Token、Cookie 或 API Key。日志在 `data/runtime/logs/` 且采用大小受限的轮转。卸载可关闭全部自动服务。
