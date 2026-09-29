"""Guard: costmap_core must never depend on ROS or on Dev 1's package."""

import ast
import pathlib
import subprocess
import sys

CORE_DIR = pathlib.Path(__file__).resolve().parent.parent / "costmap_core"

FORBIDDEN_ROOTS = {
    "rclpy",
    "rosidl_runtime_py",
    "builtin_interfaces",
    "std_msgs",
    "sensor_msgs",
    "geometry_msgs",
    "nav_msgs",
    "tf2_ros",
    "tf2_py",
    "tf2_msgs",
    "costmap_ros",
    "ugv_perception",
}


def _imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module.split(".")[0]


def test_core_sources_have_no_ros_imports():
    offenders = {
        f"{path.name}: {root}"
        for path in sorted(CORE_DIR.glob("*.py"))
        for root in _imported_roots(path)
        if root in FORBIDDEN_ROOTS
    }
    assert not offenders


def test_core_imports_with_ros_unavailable():
    # Import every core module in a fresh interpreter where ROS packages
    # cannot be found, as on a machine without ROS installed.
    modules = sorted(f"costmap_core.{p.stem}" for p in CORE_DIR.glob("*.py") if p.stem != "__init__")
    script = (
        "import importlib, sys\n"
        f"blocked = {sorted(FORBIDDEN_ROOTS)!r}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in blocked:\n"
        "            raise ImportError(f'blocked {name}')\n"
        "sys.meta_path.insert(0, Block())\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=CORE_DIR.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
