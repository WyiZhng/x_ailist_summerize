"""Install only this project's user-level macOS LaunchAgents."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIXIN_LABEL = "com.wyizhng.xaidaily.weixin-service"
TICK_LABEL = "com.wyizhng.xaidaily.daily-tick"


class LaunchdManager:
    """Generate safe plists dynamically; credentials remain in private files."""

    def __init__(
        self,
        *,
        project_root: Path = PROJECT_ROOT,
        launch_agents: Path | None = None,
        python_path: Path | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.launch_agents = launch_agents or Path.home() / "Library" / "LaunchAgents"
        # Keep a virtualenv symlink intact.  Resolving it selects the global
        # interpreter and drops the virtualenv's installed dependencies.
        self.python_path = Path(
            python_path or self.project_root / ".venv" / "bin" / "python"
        ).absolute()
        self.logs = self.project_root / "data" / "runtime" / "logs"

    def path(self, label: str) -> Path:
        return self.launch_agents / f"{label}.plist"

    def payload(self, label: str) -> dict:
        args = (
            [str(self.python_path), "-m", "app.weixin_service"]
            if label == WEIXIN_LABEL
            else [str(self.python_path), "-m", "app.daily_delivery", "--tick"]
        )
        log_name = (
            "weixin-service.log" if label == WEIXIN_LABEL else "daily-delivery.log"
        )
        value = {
            "Label": label,
            "ProgramArguments": args,
            "WorkingDirectory": str(self.project_root),
            "StandardOutPath": str(self.logs / log_name),
            "StandardErrorPath": str(self.logs / log_name),
        }
        if label == WEIXIN_LABEL:
            value["KeepAlive"] = {"SuccessfulExit": False}
        else:
            value["StartInterval"] = 300
        return value

    def install(self, *, execute: bool = True) -> None:
        if not self.python_path.is_file():
            raise RuntimeError("Configured Python executable is missing")
        self.launch_agents.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        for label in (WEIXIN_LABEL, TICK_LABEL):
            target = self.path(label)
            with target.open("wb") as handle:
                plistlib.dump(self.payload(label), handle, sort_keys=True)
            os.chmod(target, 0o600)
            subprocess.run(
                ["plutil", "-lint", str(target)], check=True, capture_output=True
            )
            if execute:
                domain = f"gui/{os.getuid()}"
                subprocess.run(
                    ["launchctl", "bootout", domain, str(target)],
                    check=False,
                    capture_output=True,
                )
                subprocess.run(
                    ["launchctl", "bootstrap", domain, str(target)],
                    check=True,
                    capture_output=True,
                )

    def uninstall(self, *, execute: bool = True) -> None:
        for label in (WEIXIN_LABEL, TICK_LABEL):
            target = self.path(label)
            if execute and target.exists():
                subprocess.run(
                    ["launchctl", "bootout", f"gui/{os.getuid()}", str(target)],
                    check=False,
                    capture_output=True,
                )
            target.unlink(missing_ok=True)

    def restart(self) -> None:
        for label in (WEIXIN_LABEL, TICK_LABEL):
            subprocess.run(
                ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{label}"],
                check=False,
                capture_output=True,
            )

    def status(self) -> str:
        return "\n".join(
            f"{label}: {'已安装' if self.path(label).exists() else '未安装'}"
            for label in (WEIXIN_LABEL, TICK_LABEL)
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage x-ai-daily LaunchAgents")
    parser.add_argument(
        "command", choices=("install", "status", "restart", "uninstall")
    )
    args = parser.parse_args(argv)
    manager = LaunchdManager()
    if args.command == "install":
        manager.install()
        print("LaunchAgents 已安装。")
    elif args.command == "uninstall":
        manager.uninstall()
        print("LaunchAgents 已卸载。")
    elif args.command == "restart":
        manager.restart()
        print("LaunchAgents 已请求重启。")
    else:
        print(manager.status())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
