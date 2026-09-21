"""Import Pandas extension modules from an extracted wheel overlay."""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path


ALLOWED_ROOT = Path("./data/CancerLncAtlas")


class PandasExtensionFinder(importlib.abc.MetaPathFinder):
    def __init__(self, overlay_root: Path) -> None:
        root = overlay_root.resolve(strict=True)
        if not root.is_dir() or root.is_symlink() or ALLOWED_ROOT not in root.parents:
            raise RuntimeError(f"Invalid Pandas extension overlay: {root}")
        modules: dict[str, Path] = {}
        for path in root.joinpath("pandas").rglob("*.so"):
            relative = path.relative_to(root).as_posix()
            stem = relative[:-3].split(".cpython-", 1)[0].split(".abi3", 1)[0]
            name = stem.replace("/", ".")
            if name in modules:
                raise RuntimeError(f"Duplicate Pandas extension module: {name}")
            modules[name] = path.resolve(strict=True)
        if not modules:
            raise RuntimeError("Pandas extension overlay contains no modules")
        self.overlay_root = root
        self.modules = modules

    def find_spec(self, fullname: str, path=None, target=None):
        source = self.modules.get(fullname)
        if source is None:
            return None
        loader = importlib.machinery.ExtensionFileLoader(fullname, str(source))
        return importlib.util.spec_from_file_location(fullname, source, loader=loader)


def install(overlay_root: str | Path) -> PandasExtensionFinder:
    root = Path(overlay_root).resolve(strict=True)
    for finder in sys.meta_path:
        if isinstance(finder, PandasExtensionFinder):
            if finder.overlay_root != root:
                raise RuntimeError("A different Pandas extension overlay is already installed")
            return finder
    finder = PandasExtensionFinder(root)
    sys.meta_path.insert(0, finder)
    return finder
