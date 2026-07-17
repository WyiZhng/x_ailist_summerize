import plistlib
from pathlib import Path

from app.launchd_manager import LaunchdManager, TICK_LABEL, WEIXIN_LABEL


def test_launchd_payloads_are_safe_and_distinct(tmp_path: Path) -> None:
    python = tmp_path / "venv" / "python"
    python.parent.mkdir()
    python.touch()
    manager = LaunchdManager(
        project_root=tmp_path, launch_agents=tmp_path / "agents", python_path=python
    )
    service = manager.payload(WEIXIN_LABEL)
    tick = manager.payload(TICK_LABEL)
    assert service["KeepAlive"] == {"SuccessfulExit": False}
    assert tick["StartInterval"] == 300 and "KeepAlive" not in tick
    assert str(python) == service["ProgramArguments"][0]
    assert "token" not in str(service).lower() and "token" not in str(tick).lower()


def test_launchd_install_and_uninstall_are_scoped(tmp_path: Path) -> None:
    python = tmp_path / "venv" / "python"
    python.parent.mkdir()
    python.touch()
    agents = tmp_path / "agents"
    manager = LaunchdManager(
        project_root=tmp_path, launch_agents=agents, python_path=python
    )
    manager.install(execute=False)
    for label in (WEIXIN_LABEL, TICK_LABEL):
        with manager.path(label).open("rb") as handle:
            assert plistlib.load(handle)["Label"] == label
    manager.uninstall(execute=False)
    assert not list(agents.glob("*.plist"))
