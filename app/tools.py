import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from typing_extensions import Annotated


class Tool(ABC):

    name: str
    description: str

    @abstractmethod
    def execute(self) -> Any:
        pass


class ReadTool(Tool):

    name = "Read"
    description = "Read and return the contents of a file"

    def __init__(
        self,
        file_path: Annotated[str, {
            "type": "string",
            "description": "The path to the file to read",
        }],
    ):
        self.file_path = file_path

    def execute(self) -> str:
        try:
            with open(self.file_path, "r") as fd:
                return fd.read()
        except Exception as error:
            return f"Error reading file {self.file_path}: {error}"


class WriteTool(Tool):

    name = "Write"
    description = "Write content to a file"

    def __init__(
        self,
        file_path: Annotated[str, {
            "type": "string",
            "description": "The path of the file to write to",
        }],
        content: Annotated[str, {
            "type": "string",
            "description": "The content to write to the file",
        }],
    ):
        self.file_path = file_path
        self.content = content

    def execute(self) -> str:
        try:
            with open(self.file_path, "w") as fd:
                fd.write(self.content)

            return f"Successfully wrote {len(self.content)} characters to {self.file_path}"
        except Exception as error:
            return f"Error writing to file {self.file_path}: {error}"


class Toolbox:

    def __init__(self):
        self.tool_class_by_name: Dict[str, Type[Tool]] = {}
        self.tool_schemas: List[Any] = []

    def add(self, tool_class: Type[Tool]):
        self.tool_class_by_name[tool_class.name] = tool_class

        init_signature = inspect.signature(tool_class.__init__)
        parameters = list(init_signature.parameters.values())[1:]  # skip 'self'

        parameter_schemas: Dict[str, Any] = {}
        for parameter in parameters:
            if parameter.annotation is inspect.Parameter.empty:
                raise ValueError(f"parameter {parameter.name} of tool {tool_class.name} is missing type annotation")

            if not hasattr(parameter.annotation, "__metadata__"):
                raise ValueError(f"parameter {parameter.name} of tool {tool_class.name} is missing metadata in type annotation")

            metadata = parameter.annotation.__metadata__[0]
            parameter_schemas[parameter.name] = {
                "type": metadata["type"],
                "description": metadata["description"],
            }

        self.tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool_class.name,
                    "description": tool_class.description,
                    "parameters": {
                        "type": "object",
                        "properties": parameter_schemas,
                        "required": list(parameter_schemas.keys()),
                    },
                },
            }
        )

    def use(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        tool_class = self.tool_class_by_name.get(tool_name)
        if tool_class is None:
            raise ValueError(f"tool {tool_name} not found in toolbox")

        tool_instance = tool_class(**arguments)
        return tool_instance.execute()
