import json
import subprocess
import sys
from pathlib import Path


def read_json(path):
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
        return subprocess.check_output(["git", *args], text=True).strip()
    state = {"commit": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"), "config": config,
             "argv": sys.argv, "python": sys.executable, "cwd": str(Path.cwd())}
    write_json(output / "run.json", state)
    diff = git("diff", "HEAD", "--", ".")
    if diff:
        (output / "working-tree.patch").write_text(diff + "\n", encoding="utf-8")
    # Untracked implementation files are copied too: a commit alone would omit them.
    for name in git("ls-files", "--others", "--exclude-standard").splitlines():
        path = Path(name)
        if path.is_file() and path.suffix in {".py", ".json", ".txt", ".toml"}:
            target = output / "untracked" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return state
