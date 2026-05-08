"""Collect repo-backed metrics for the sensPy Sensometrics poster."""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 poster builds
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from senspy.core.types import Protocol

POSTER = ROOT / "poster"
CHART_DATA = POSTER / "chart_data"
ASSETS = POSTER / "assets"
CHART_DATA.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

DISPLAY_NAMES = {
    "duotrio": "Duo-trio",
    "triangle": "Triangle",
    "twoafc": "2-AFC",
    "threeafc": "3-AFC",
    "tetrad": "Tetrad",
    "hexad": "Hexad",
    "twofive": "2-out-of-5",
    "twofivef": "2-out-of-5F",
}


def protocol_guess(protocol: str) -> float:
    try:
        return Protocol(protocol).p_guess
    except ValueError:
        print(f"[collect] warning: unknown protocol {protocol!r}; using p_guess=0")
        return 0.0


def display_name(protocol: str) -> str:
    if protocol not in DISPLAY_NAMES:
        print(f"[collect] warning: missing display name for protocol {protocol!r}")
    return DISPLAY_NAMES.get(protocol, protocol)


def parse_protocols() -> list[str]:
    return [protocol.value for protocol in Protocol]


def parse_double_protocols() -> list[str]:
    tree = ast.parse((ROOT / "senspy" / "links" / "double.py").read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("double_") and node.name.endswith("_link"):
                names.append(node.name.removeprefix("double_").removesuffix("_link"))
    return sorted(names)


def estimate_pytest_items(node: ast.FunctionDef, source: Path) -> int:
    multiplier = 1
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        if isinstance(func, ast.Attribute) and func.attr == "parametrize":
            if len(dec.args) < 2:
                continue
            try:
                values = ast.literal_eval(dec.args[1])
            except (TypeError, ValueError):
                print(
                    "[collect] warning: cannot statically count "
                    f"{source.name}:{node.lineno} parametrization; using 1"
                )
                continue
            try:
                multiplier *= len(values)
            except TypeError:
                print(
                    "[collect] warning: parametrization has no length in "
                    f"{source.name}:{node.lineno}; using 1"
                )
                pass
    return multiplier


def test_inventory() -> tuple[list[dict[str, object]], int, int]:
    rows: list[dict[str, object]] = []
    total_functions = 0
    total_items = 0

    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        tree = ast.parse(path.read_text())
        functions = 0
        estimated = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                functions += 1
                estimated += estimate_pytest_items(node, path)
        total_functions += functions
        total_items += estimated
        category = path.stem.removeprefix("test_").replace("_", " ")
        rows.append(
            {
                "file": path.name,
                "category": category,
                "test_functions": functions,
                "estimated_collected_items": estimated,
            }
        )

    return rows, total_functions, total_items


def pytest_collected_count() -> int | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        print(f"[collect] warning: could not run pytest collection: {exc}")
        return None
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        print(f"[collect] warning: pytest collection failed: {detail}")
        return None
    match = re.search(r"=+\s+(\d+)\s+tests collected", result.stdout)
    if match:
        return int(match.group(1))
    print("[collect] warning: pytest collection count not found; using AST estimate")
    return None


def exported_api_count() -> tuple[int, list[str]]:
    tree = ast.parse((ROOT / "senspy" / "__init__.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    values = ast.literal_eval(node.value)
                    return len(values), values
    return 0, []


def dataclass_inventory() -> list[str]:
    names: list[str] = []
    for path in sorted((ROOT / "senspy").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Name) and target.id == "dataclass":
                    names.append(node.name)
                    break
                if isinstance(target, ast.Attribute) and target.attr == "dataclass":
                    names.append(node.name)
                    break
    return sorted(set(names))


def write_protocol_coverage(protocols: list[str], doubles: list[str]) -> None:
    with (CHART_DATA / "protocol_coverage.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["protocol", "display", "single", "double", "p_guess"]
        )
        writer.writeheader()
        for protocol in protocols:
            writer.writerow(
                {
                    "protocol": protocol,
                    "display": display_name(protocol),
                    "single": 1,
                    "double": 1 if protocol in doubles else 0,
                    "p_guess": round(protocol_guess(protocol), 4),
                }
            )


def write_test_inventory(rows: list[dict[str, object]]) -> None:
    with (CHART_DATA / "test_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "category",
                "test_functions",
                "estimated_collected_items",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def make_qr() -> None:
    try:
        import qrcode
    except ImportError:
        print("[collect] warning: qrcode package not found; skipping QR generation")
        return

    qr = qrcode.QRCode(border=2, box_size=12)
    qr.add_data("https://github.com/aigorahub/sensPy")
    qr.make(fit=True)
    img = qr.make_image(fill_color="#17291f", back_color="#f4f0e6")
    img.save(ASSETS / "qr-senspy-github.png")


def main() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package = pyproject["tool"]["poetry"]

    protocols = parse_protocols()
    doubles = parse_double_protocols()
    tests, test_functions, estimated_items = test_inventory()
    collected_items = pytest_collected_count()
    api_count, api_exports = exported_api_count()
    dataclasses = dataclass_inventory()

    fixture_path = ROOT / "tests" / "fixtures" / "golden_sensr.json"
    fixture = json.loads(fixture_path.read_text())
    metadata = fixture.get("metadata", {})

    write_protocol_coverage(protocols, doubles)
    write_test_inventory(tests)
    make_qr()

    summary = {
        "package": package["name"],
        "version": package["version"],
        "description": package["description"],
        "sensr_version": metadata.get("sensR_version", "unknown"),
        "r_version": metadata.get("R_version", "unknown"),
        "single_protocol_count": len(protocols),
        "double_protocol_count": len(doubles),
        "total_protocol_variants": len(protocols) + len(doubles),
        "protocols": protocols,
        "double_protocols": doubles,
        "test_functions": test_functions,
        "estimated_pytest_items": estimated_items,
        "collected_pytest_items": collected_items or estimated_items,
        "api_export_count": api_count,
        "api_exports": api_exports,
        "dataclass_count": len(dataclasses),
        "dataclasses": dataclasses,
        "poster_url": "https://github.com/aigorahub/sensPy",
    }

    (CHART_DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "[collect] "
        f"{summary['total_protocol_variants']} variants, "
        f"{test_functions} test functions, "
        f"{summary['collected_pytest_items']} collected pytest items"
    )


if __name__ == "__main__":
    main()
