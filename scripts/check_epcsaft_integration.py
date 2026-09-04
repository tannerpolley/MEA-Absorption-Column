from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CONTRACT_PATH = REPO_ROOT / "integration" / "epcsaft_contract.json"
IMPORT_PATTERN = re.compile(r"^\s*(?:from\s+epcsaft(?:\.[\w.]+)?\s+import\b|import\s+epcsaft\b)")


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _distribution(name: str):
    try:
        return importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _direct_url_payload(dist) -> dict[str, Any] | None:
    if dist is None:
        return None
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    return json.loads(text)


def _local_wheel_identity(direct_url: dict[str, Any] | None) -> dict[str, str] | None:
    if not direct_url or not str(direct_url.get("url", "")).startswith("file:"):
        return None
    parsed = urlparse(str(direct_url["url"]))
    wheel_path = Path(unquote(parsed.path)).resolve()
    if not wheel_path.is_file():
        raise RuntimeError(f"Installed ePC-SAFT wheel source no longer exists: {wheel_path}")
    digest = hashlib.sha256()
    with wheel_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"wheel_path": str(wheel_path), "wheel_sha256": digest.hexdigest()}


def resolve_epcsaft(contract: dict[str, Any]) -> dict[str, Any]:
    thermo_models = importlib.import_module("mea_absorption_column.Thermodynamics.thermo_models")
    thermo_models.ensure_epcsaft_importable()
    epcsaft = importlib.import_module("epcsaft")
    dist = _distribution(contract["package"]["name"])
    direct_url = _direct_url_payload(dist)
    local_wheel = _local_wheel_identity(direct_url)
    module_path = Path(epcsaft.__file__).resolve()
    version = importlib.metadata.version(contract["package"]["name"])

    source_kind = "release"
    source_detail = str(module_path)
    if direct_url and "vcs_info" in direct_url:
        source_kind = "pinned_git"
        source_detail = json.dumps(direct_url, sort_keys=True)
    elif direct_url and direct_url.get("url"):
        source_kind = "local_file" if str(direct_url["url"]).startswith("file:") else "direct_url"
        source_detail = json.dumps(direct_url, sort_keys=True)

    resolved = {
        "module_path": str(module_path),
        "version": version,
        "source_kind": source_kind,
        "source_detail": source_detail,
    }
    if local_wheel is not None:
        resolved.update(local_wheel)
    return resolved


def _iter_python_files(scan_roots: list[str]):
    for root in scan_roots:
        path = REPO_ROOT / root
        if not path.exists():
            continue
        yield from path.rglob("*.py")


def _is_import_module_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return isinstance(func.value, ast.Name) and func.value.id == "importlib" and func.attr == "import_module"
    return isinstance(func, ast.Name) and func.id == "import_module"


def _is_spec_from_file_location_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return (
            isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "importlib"
            and func.value.attr == "util"
            and func.attr == "spec_from_file_location"
        )
    return False


def _string_arg(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def scan_direct_imports(contract: dict[str, Any]) -> list[str]:
    allowlist = {entry.replace("\\", "/") for entry in contract.get("allowed_direct_import_paths", [])}
    private_exceptions = {entry.replace("\\", "/") for entry in contract.get("private_import_exception_paths", [])}
    forbidden_prefixes = tuple(contract.get("forbidden_import_prefixes", []))
    failures: list[str] = []
    for path in _iter_python_files(contract.get("scan_roots", [])):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            failures.append(f"{rel}:{exc.lineno}: could not parse Python for import scanning")
            continue

        if rel not in private_exceptions:
            for node in ast.walk(tree):
                module_name = None
                line_no = getattr(node, "lineno", 1)
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(alias.name.startswith(prefix) for prefix in forbidden_prefixes):
                            module_name = alias.name
                            break
                elif isinstance(node, ast.ImportFrom):
                    if node.module and any(node.module.startswith(prefix) for prefix in forbidden_prefixes):
                        module_name = node.module
                elif isinstance(node, ast.Call) and (_is_import_module_call(node) or _is_spec_from_file_location_call(node)):
                    candidate = _string_arg(node)
                    if candidate and any(candidate.startswith(prefix) for prefix in forbidden_prefixes):
                        module_name = candidate

                if module_name is not None:
                    failures.append(f"{rel}:{line_no}: uses forbidden private epcsaft module {module_name}")
                    break

        if rel in allowlist:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if IMPORT_PATTERN.search(line):
                failures.append(f"{rel}:{line_no}: direct epcsaft import outside allowlist")
    return failures


def run_smoke(contract: dict[str, Any]) -> dict[str, Any]:
    thermo_models = importlib.import_module(contract["smoke"]["module"])
    thermo_models.ensure_epcsaft_importable()
    dataset_path = Path(thermo_models.MEA_THERMODYNAMICS_EPCSAFT_DATASET)
    if not dataset_path.exists():
        raise RuntimeError(f"Expected vendored or configured dataset path to exist: {dataset_path}")
    diagnostics = thermo_models.epcsaft_state_contribution_diagnostics(
        323.15,
        109500.0,
        (1.0e-8, 0.055, 0.888, 0.028, 0.027, 0.001),
        phase="liquid",
        mixture_kind="ionic",
    )
    if diagnostics["phi_co2"] <= 0.0:
        raise RuntimeError("The ionic ePC-SAFT smoke state returned a nonpositive CO2 fugacity coefficient.")
    reactive = importlib.import_module("mea_absorption_column.Thermodynamics.reactive_bundle")
    # Pin the accepted input/export combination to this runtime, without rewriting its provenance.
    for name, expected in contract["final_identity"]["reactive_inputs_sha256"].items():
        if hashlib.sha256((reactive.DATASET / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Selected reactive input differs from the integration pin: {name}")
    reactive_result = reactive.reactive_liquid().solve(
        313.15, 109500.0, [0.03, 0.11, 0.86]
    )
    return {
        "dataset": str(dataset_path),
        "parameter_fingerprint": diagnostics["parameter_fingerprint"],
        "phi_co2": diagnostics["phi_co2"],
        "reactive_parameter_fingerprint": reactive_result["parameter_fingerprint"],
        "reactive_density_mol_m3": reactive_result["density_mol_m3"],
    }


def _version_triplet(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"Unsupported version string: {value!r}")
    return tuple(int(part) for part in match.groups())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check MEA-Absorption-Column ePC-SAFT integration contract.")
    parser.add_argument("--mode", choices=("stable", "dev", "final"), default=None, help="Select the package source lane.")
    parser.add_argument("--self-only", action="store_true", help="Skip the repo-specific smoke step outside final mode.")
    args = parser.parse_args(argv)

    contract = load_contract()
    mode = args.mode or contract["package"].get("default_mode", "stable")
    resolved = resolve_epcsaft(contract)
    allowed_sources = set(contract["package"]["allowed_sources"][mode])
    errors: list[str] = []

    if resolved["source_kind"] not in allowed_sources:
        errors.append(
            f"Resolved epcsaft source kind {resolved['source_kind']!r} from {resolved['source_detail']} "
            f"is not allowed for {mode} mode."
        )

    if mode == "final":
        identity = contract["final_identity"]
        if Path(resolved.get("wheel_path", "")).name != identity["wheel_filename"]:
            errors.append(
                f"Resolved ePC-SAFT wheel filename {Path(resolved.get('wheel_path', '')).name!r} "
                f"does not match frozen identity {identity['wheel_filename']!r}."
            )
        if resolved.get("wheel_sha256") != identity["wheel_sha256"]:
            errors.append(
                f"Resolved ePC-SAFT wheel SHA-256 {resolved.get('wheel_sha256')!r} "
                f"does not match frozen identity {identity['wheel_sha256']!r}."
            )

    try:
        version = _version_triplet(str(resolved["version"]))
        minimum = _version_triplet(contract["package"]["minimum_version"])
        if version < minimum:
            errors.append(
                f"Resolved epcsaft version {resolved['version']} is below minimum {contract['package']['minimum_version']}."
            )
    except Exception:
        errors.append(f"Could not compare resolved epcsaft version {resolved['version']!r}.")

    epcsaft = importlib.import_module("epcsaft")
    for symbol in contract.get("required_public_symbols", []):
        if not hasattr(epcsaft, symbol):
            errors.append(f"Missing required public symbol: {symbol}")

    errors.extend(scan_direct_imports(contract))

    smoke_payload = None
    if mode == "final" or not args.self_only:
        try:
            smoke_payload = run_smoke(contract)
        except Exception as exc:
            errors.append(f"Smoke check failed: {exc}")

    print(f"contract mode: {mode}")
    print(f"epcsaft module path: {resolved['module_path']}")
    print(f"epcsaft version: {resolved['version']}")
    print(f"epcsaft source kind: {resolved['source_kind']}")
    print(f"epcsaft source detail: {resolved['source_detail']}")
    if resolved.get("wheel_sha256"):
        print(f"epcsaft wheel SHA-256: {resolved['wheel_sha256']}")
    if mode == "final":
        print(f"Engine commit: {contract['final_identity']['engine_commit']}")
    if smoke_payload is not None:
        print(f"dataset path: {smoke_payload['dataset']}")
        print(f"parameter fingerprint: {smoke_payload['parameter_fingerprint']}")
        print(f"CO2 fugacity coefficient: {smoke_payload['phi_co2']}")
        print(f"Nine-species parameter fingerprint: {smoke_payload['reactive_parameter_fingerprint']}")
        print(f"Nine-species liquid density [mol/m3]: {smoke_payload['reactive_density_mol_m3']}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
