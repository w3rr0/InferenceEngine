"""Expose generated stubs under the package's public API."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).parents[1] / "inference_engine"
BINDINGS_DIR = PACKAGE_DIR / "_bindings"


def main() -> None:
    for binding_stub in BINDINGS_DIR.glob("*.pyi"):
        if binding_stub.name == "__init__.pyi":
            continue

        module_name = binding_stub.stem
        public_stub = PACKAGE_DIR / binding_stub.name
        public_stub.write_text(
            f"from ._bindings.{module_name} import *\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
