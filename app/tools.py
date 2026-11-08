import os
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .lsp_client import LSPClient
from .skills import Skillbox

if TYPE_CHECKING:
    from .agent import Agent


@dataclass
class ToolParameter:

    name: Union[str, Tuple[str, str]]
    type: str
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    minimum: Optional[int] = None
    required: bool = True


class Tool(ABC):

    name: str
    description: str
    parameters: List[ToolParameter]

    @abstractmethod
    def execute(self, agent: "Agent", **kwargs: Any) -> Any:
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

    def execute(self, agent: "Agent", file_path: str, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
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

    def execute(self, agent: "Agent", file_path: str, content: str, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
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

    def execute(self, agent: "Agent", command: str, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
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

    def execute(self, agent: "Agent", name: str, args: Optional[str] = None, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        skill = self.skillbox.get(name)
        if skill is None:
            return f"Skill '{name}' not found"

        child = agent.fork(name="sub-agent")

        arguments = shlex.split(args) if args else []
        child.push_message({"role": "user", "content": skill.format(arguments, include_location=False)})

        return child.loop()


class LspTool(Tool):

    name = "LSP"
    description = "Returns the file, line, and character where a symbol is defined."
    parameters = [
        ToolParameter(
            name="operation",
            type="string",
            enum=["goToDefinition"],
        ),
        ToolParameter(
            name=("file_path", "filePath"),
            type="string",
            description="Absolute path, or a path relative to the agent's current working directory.",
        ),
        ToolParameter(
            name="line",
            type="integer",
            minimum=1,
            description="1-based line number of the symbol.",
        ),
        ToolParameter(
            name="character",
            type="integer",
            minimum=1,
            description="1-based character offset in UTF-16 code units (same as character count for ASCII).",
        )
    ]

    def __init__(self, client: LSPClient):
        self.client = client

    def execute(self, agent: "Agent", operation: str, file_path: str, line: int, character: int, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return f"File '{file_path}' not found"

        if operation == "goToDefinition":
            with open(file_path, "r", encoding="utf-8") as fd:
                text = fd.read()

            uri = f"file://{file_path}"
            extension = os.path.splitext(file_path)[1]

            self.client.notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri,
                    "languageId": self.client.extension_to_language.get(extension, ""),
                    "version": 1,
                    "text": text,
                }
            })

            response = self.client.request("textDocument/definition", {
                "textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": character - 1}
            })

            return f"Defined in {response['uri']}, from {self._to_range_string(response['range']['start'])} to {self._to_range_string(response['range']['end'])}"

        return "LSP is not implemented yet"

    def _to_range_string(self, range: Dict[str, Any]) -> str:
        return f"{range['line'] + 1}:{range['character'] + 1}"


class Toolbox:

    def __init__(self):
        self.tool_by_name: Dict[str, Tool] = {}
        self.tool_schemas: List[Any] = []

    def add(self, tool: Tool):
        self.tool_by_name[tool.name] = tool

        parameter_schemas: Dict[str, Any] = {}
        required_parameters: List[str] = []

        for parameter in tool.parameters:
            if isinstance(parameter.name, tuple):
                name = parameter.name[1]
            else:
                name = parameter.name

            parameter_schemas[name] = {
                key: value
                for key, value in {  # pyright: ignore[reportUnknownVariableType]
                    "type": parameter.type,
                    "enum": parameter.enum,
                    "minimum": parameter.minimum,
                    "description": parameter.description,
                }.items()
                if value is not None
            }

            if parameter.required:
                required_parameters.append(name)

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

    def use(self, agent: "Agent", tool_name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.tool_by_name.get(tool_name)
        if tool is None:
            raise ValueError(f"tool {tool_name} not found in toolbox")

        remapped_arguments: Dict[str, Any] = {}
        for parameter in tool.parameters:
            if isinstance(parameter.name, tuple):
                python_name, original_name = parameter.name
            else:
                python_name = original_name = parameter.name

            remapped_arguments[python_name] = arguments[original_name]

        return tool.execute(agent, **remapped_arguments)
