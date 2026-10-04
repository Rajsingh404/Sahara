"""PlatformIO pre-build check: include/sound_classes.h must list the classes in
the same order as SOUND_CLASSES in the repo's src/config.py."""
import ast
import re
from pathlib import Path

Import("env")  # noqa: F821  (provided by PlatformIO)

project_dir = Path(env.subst("$PROJECT_DIR"))  # noqa: F821
config_py = project_dir.parents[1] / "src" / "config.py"
header = project_dir / "include" / "sound_classes.h"

if not config_py.exists():
    print(f"check_classes: {config_py} not found, skipping class order check")
else:
    tree = ast.parse(config_py.read_text())
    expected = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SOUND_CLASSES" for t in node.targets
        ):
            expected = [ast.literal_eval(e) for e in node.value.elts]
    found = re.findall(r"SOUND_(\w+)\s*=\s*(\d+)", header.read_text())
    actual = [name.lower() for name, idx in sorted(found, key=lambda x: int(x[1])) if name != "CLASS_COUNT"]
    if expected != actual:
        raise SystemExit(
            f"check_classes: class order mismatch\n  src/config.py: {expected}\n  sound_classes.h: {actual}"
        )
    print(f"check_classes: {len(actual)} classes match src/config.py")
