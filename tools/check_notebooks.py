"""Validate notebook source without running training; optionally use fresh kernels."""
from argparse import ArgumentParser
from pathlib import Path
import ast
import importlib.util
import json
import re


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute inspection mode using nbclient")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    schema_available = importlib.util.find_spec("nbformat") is not None
    paths = sorted((root / "notebooks").glob("*.ipynb"))
    for path in paths:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        identities = set()
        for cell in notebook["cells"]:
            identity = cell["id"]
            assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity) and identity not in identities
            identities.add(identity)
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [] and cell["execution_count"] is None
                ast.parse("".join(cell["source"]), filename=path.name)
        if schema_available:
            import nbformat
            nbformat.validate(nbformat.from_dict(notebook))
        if args.execute:
            import nbformat
            from nbclient import NotebookClient
            NotebookClient(nbformat.from_dict(notebook), timeout=180, kernel_name="python3",
                           resources={"metadata": {"path": str(root)}}).execute()
        print(f"PASS {path.name}")
    for path in [root / "README.md", *(root / "docs").glob("*.md"), root / "data/README.md"]:
        for target in re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", path.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)
    print("Documentation links checked.")
    if not schema_available:
        print("nbformat unavailable: structural checks only; install the dev extra for schema validation.")


if __name__ == "__main__":
    main()
