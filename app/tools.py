import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .skills import Skillbox


@dataclass
class ToolParameter:

    name: str
    type: str
    description: str
    required: bool = True


class Tool(ABC):

    name: str
    description: str
    parameters: List[ToolParameter]

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        pass


class ReadTool(Tool):

    name = "Read"
    description = "Read and return the contents of a file"
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="The path to the file to read"
        )
    ]

    def execute(self, file_path: str, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        try:
            with open(file_path, "r") as fd:
                return fd.read()
        except Exception as error:
            return f"Error reading file {file_path}: {error}"


class WriteTool(Tool):

    name = "Write"
    description = "Write content to a file"
    parameters = [
        ToolParameter(
            name="file_path",
            type="string",
            description="The path of the file to write to"
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The content to write to the file"
        )
    ]

    def execute(self, file_path: str, content: str, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        try:
            with open(file_path, "w") as fd:
                fd.write(content)

            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as error:
            return f"Error writing to file {file_path}: {error}"


class BashTool(Tool):

    name = "Bash"
    description = "Execute a shell command"
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute"
        )
    ]

    def execute(self, command: str, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        import subprocess

        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Error executing command {command}: {result.stderr.strip()}"

            return result.stdout.strip()
        except Exception as error:
            return f"Error executing command {command}: {error}"


class SkillTool(Tool):

    name = "Skill"
    description = "Load a skill's instructions into the conversation"
    parameters = [
        ToolParameter(
            name="name",
            type="string",
            description="The name of the skill to use"
        ),
        ToolParameter(
            name="args",
            type="string",
            description="Optional arguments for the skill",
            required=False,
        )
    ]

    def __init__(
        self,
        skillbox: Skillbox,
    ):
        self.skillbox = skillbox

    def execute(self, name: str, args: Optional[str] = None, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        skill = self.skillbox.get(name)
        if skill is None:
            return f"Skill '{name}' not found"

        arguments = shlex.split(args) if args else []
        return skill.format(arguments)


class Toolbox:

    def __init__(self):
        self.tool_by_name: Dict[str, Tool] = {}
        self.tool_schemas: List[Any] = []

    def add(self, tool: Tool):
        self.tool_by_name[tool.name] = tool

        parameter_schemas: Dict[str, Any] = {}
        required_parameters: List[str] = []

        for parameter in tool.parameters:
            parameter_schemas[parameter.name] = {
                "type": parameter.type,
                "description": parameter.description,
            }

            if parameter.required:
                required_parameters.append(parameter.name)

        self.tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": parameter_schemas,
                        "required": required_parameters,
                    },
                },
            }
        )

    def use(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.tool_by_name.get(tool_name)
        if tool is None:
            raise ValueError(f"tool {tool_name} not found in toolbox")

        return tool.execute(**arguments)
