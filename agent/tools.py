from abc import ABC, abstractmethod
from typing import Any
import textwrap


class BaseTool(ABC):
    name: str

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class Tool(BaseTool):
    name: str
    description: str
    inputs: dict
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Implement forward() in your Tool subclass.")

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()
        return self.forward(*args, **kwargs)

    def setup(self):
        self.is_initialized = True

    def to_code_prompt(self) -> str:
        args_signature = ", ".join(f"{k}: {v['type']}" for k, v in self.inputs.items())
        tool_doc = self.description
        if self.inputs:
            args_doc = "Args:\n" + textwrap.indent(
                "\n".join(f"{k}: {v['description']}" for k, v in self.inputs.items()), "    "
            )
            tool_doc += f"\n\n{args_doc}"
        return f'def {self.name}({args_signature}) -> {self.output_type}:\n    """{tool_doc}\n    """'

    def to_tool_calling_prompt(self) -> str:
        return f"{self.name}: {self.description}\n    Takes inputs: {self.inputs}\n    Returns an output of type: {self.output_type}"


__all__ = ["Tool"]
