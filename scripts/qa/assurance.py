"""Compatibility entry point for the maintained package implementation."""

from importlib import import_module

_implementation = import_module("fpl_assistant.qa.assurance")

# Re-export public and private helpers because historical callers and tests
# import a small number of underscored functions from the scripts namespace.
globals().update(
    {
        name: value
        for name, value in vars(_implementation).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)


if __name__ == "__main__":
    _implementation.main()
