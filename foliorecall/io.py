import json
import subprocess
import sys
from pathlib import Path


def read_json(path):
    if not Path(path).is_file():
        raise ValueError(f"配置或数据文件不存在: {path}；安装态请显式指定配置路径")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def provenance(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def git(*args):
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    from . import __version__
    state = {"commit": None, "status": None, "config": config, "package_version": __version__,
             "argv": sys.argv, "python": sys.executable, "cwd": str(Path.cwd())}
    # Only the checkout containing this implementation can identify its code.
    try:
        root = Path(git("rev-parse", "--show-toplevel")).resolve()
        if not Path(__file__).resolve().is_relative_to(root):
            raise ValueError("installed package is outside the current Git checkout")
        state.update(commit=git("rev-parse", "HEAD"), status=git("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        state["git_unavailable"] = str(exc)
        write_json(output / "run.json", state)
        return state
    write_json(output / "run.json", state)
    diff = git("diff", "HEAD", "--", ".")
    if diff:
        (output / "working-tree.patch").write_text(diff + "\n", encoding="utf-8")
    # Untracked implementation files are copied too: a commit alone would omit them.
    for name in git("ls-files", "--others", "--exclude-standard").splitlines():
        path = root / name
        if path.is_file() and path.suffix in {".py", ".json", ".txt", ".toml"}:
            target = output / "untracked" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return state
