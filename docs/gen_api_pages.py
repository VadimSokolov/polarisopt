"""Auto-generate API reference markdown stubs from the source tree.

Runs as a mkdocs-gen-files script: for every Python module under
``src/polarisopt/``, emit a stub like

    # polarisopt.design.lhs

    ::: polarisopt.design.lhs

so mkdocstrings can render the API documentation. A ``SUMMARY.md`` is
generated so mkdocs-literate-nav can build the navigation.

Internal modules (leading underscore) are skipped.
"""

from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

SRC_ROOT = Path("src/polarisopt")
NAV_FILE = Path("reference/api/SUMMARY.md")

nav = mkdocs_gen_files.Nav()


def _module_name(path: Path) -> str:
    parts = path.with_suffix("").relative_to(SRC_ROOT.parent).parts
    return ".".join(parts)


def _doc_path(path: Path) -> Path:
    rel = path.with_suffix(".md").relative_to(SRC_ROOT)
    return Path("reference/api") / rel


for source in sorted(SRC_ROOT.rglob("*.py")):
    if any(part.startswith("_") and part != "__init__.py" for part in source.relative_to(SRC_ROOT).parts):
        # Skip private modules like _mock_runner.py
        continue
    parts = source.relative_to(SRC_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            # The top-level polarisopt __init__
            doc_path = Path("reference/api/index.md")
            module = "polarisopt"
            nav_parts: tuple[str, ...] = ("polarisopt",)
        else:
            doc_path = Path("reference/api") / Path(*parts) / "index.md"
            module = ".".join(("polarisopt",) + parts)
            nav_parts = ("polarisopt",) + parts
    else:
        doc_path = Path("reference/api") / Path(*parts).with_suffix(".md")
        module = ".".join(("polarisopt",) + parts)
        nav_parts = ("polarisopt",) + parts

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        fd.write(f"# `{module}`\n\n::: {module}\n")
    mkdocs_gen_files.set_edit_path(doc_path, source)
    nav[nav_parts] = doc_path.relative_to("reference/api").as_posix()

with mkdocs_gen_files.open(NAV_FILE, "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
