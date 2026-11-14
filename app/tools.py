import os
import shlex
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

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

    def collect_async_message(self, agent: "Agent") -> Optional[str]:
        return None


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

    def __init__(
        self,
        before_open: Optional[Callable[[str], Any]] = None,
        after_write: Optional[Callable[[str, str], Any]] = None,
    ):
        self.before_open = before_open
        self.after_write = after_write

    def execute(self, agent: "Agent", file_path: str, content: str, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        if self.before_open:
            self.before_open(file_path)

        try:
            with open(file_path, "w") as fd:
                fd.write(content)
        except Exception as error:
            return f"Error writing to file {file_path}: {error}"

        # Notify listeners (e.g. the LSP) first, so diagnostics for the *new* content can be collected below.
        if self.after_write:
            self.after_write(file_path, content)

        message = f"Successfully wrote {len(content)} characters to {file_path}"

        async_messages = agent.toolbox.collect_async_messages(agent)
        if async_messages:
            message += "\n\n" + "\n".join(async_messages)

        return message


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
            result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode != 0:
                return f"Error executing command {command} (exit code {result.returncode}): {result.stdout.strip()}"

            return result.stdout.strip() or "<no output>"
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
            enum=["goToDefinition", "findReferences"],
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

        self._handlers = {
            "goToDefinition": self.go_to_definition,
            "findReferences": self.find_references,
        }

        self._file_versions: Dict[str, int] = defaultdict(int)
        self._notifications_cursor = 0
        self._pending_notifications: List[Dict[str, Any]] = []

    def execute(self, agent: "Agent", operation: str, file_path: str, line: int, character: int, **kwargs: Any) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return f"File '{file_path}' not found"

        handler = self._handlers.get(operation)
        if handler:
            return handler(file_path, line, character)

        return "LSP is not implemented yet"

    def collect_async_message(self, agent: "Agent"):
        return self.collect_diagnostics(agent)

    def _to_uri(self, file_path: str) -> str:
        return f"file://{os.path.abspath(file_path)}"

    def notify_open(self, file_path: str) -> Tuple[str, int]:
        uri = self._to_uri(file_path)
        version = self._file_versions[uri]

        if version != 0:
            return uri, version

        with open(file_path, "r", encoding="utf-8") as fd:
            text = fd.read()

        extension = os.path.splitext(file_path)[1]
        version = self._file_versions[uri] = 1

        self.client.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": self.client.extension_to_language.get(extension, ""),
                "version": version,
                "text": text,
            }
        })

        return uri, version

    def notify_changed(self, file_path: str, content: str) -> Tuple[str, int]:
        uri = self._to_uri(file_path)
        version = self._file_versions[uri]

        if version == 0:
            return self.notify_open(file_path)

        version += 1
        self._file_versions[uri] = version

        self.client.notify("textDocument/didChange", {
            "textDocument": {
                "uri": uri,
                "version": version,
            },
            "contentChanges": [
                {
                    "text": content
                }
            ]
        })

        return uri, version

    def on_file_written(self, file_path: str, content: str):
        uri = self._to_uri(file_path)

        # Anything already received for this file describes the old content: drop it.
        self._fetch_notifications(0)
        self._pending_notifications = [
            notification
            for notification in self._pending_notifications
            if not self._is_diagnostics_for(notification, uri)
        ]

        uri, version = self.notify_changed(file_path, content)

        extension = os.path.splitext(file_path)[1]
        if extension in self.client.extension_to_language:
            self.wait_for_diagnostics(uri, version)

    def _fetch_notifications(self, timeout_ms: int = 0) -> List[Dict[str, Any]]:
        response = self.client.notifications(self._notifications_cursor, timeout_ms)
        self._notifications_cursor = response["cursor"]

        notifications = response.get("notifications") or []
        self._pending_notifications.extend(notifications)

        return notifications

    def _is_diagnostics_for(self, notification: Dict[str, Any], uri: str, version: Optional[int] = None) -> bool:
        if notification.get("method") != "textDocument/publishDiagnostics":
            return False

        params = notification.get("params") or {}
        if params.get("uri") != uri:
            return False

        notification_version = params.get("version")
        return version is None or notification_version is None or notification_version == version

    def wait_for_diagnostics(self, uri: str, version: int, timeout_seconds: float = 5.0) -> bool:
        """Block until the server publishes diagnostics for `uri` at `version`, or the timeout expires."""

        deadline = time.monotonic() + timeout_seconds

        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                return False

            notifications = self._fetch_notifications(remaining_ms)
            if any(self._is_diagnostics_for(notification, uri, version) for notification in notifications):
                return True

            if not notifications:
                time.sleep(0.05)  # in case the endpoint does not honor the timeout

    def collect_diagnostics(self, agent: "Agent"):
        self._fetch_notifications(0)
        notifications, self._pending_notifications = self._pending_notifications, []

        # publishDiagnostics always carries the *full* set for a file: the latest one wins.
        latest_per_uri: Dict[str, List[Dict[str, Any]]] = {}

        for notification in notifications:
            if notification.get("method") != "textDocument/publishDiagnostics":
                continue

            params = notification["params"]

            uri = params["uri"]
            version = params.get("version")
            if version is not None and self._file_versions.get(uri) != version:
                continue

            latest_per_uri[uri] = params.get("diagnostics") or []

        diagnostics_per_uri = {uri: diagnostics for uri, diagnostics in latest_per_uri.items() if diagnostics}
        if not diagnostics_per_uri:
            return None

        reply = StringIO()
        reply.write("<new-diagnostics>The following new diagnostic issues were detected:\n")

        for uri, diagnostics in diagnostics_per_uri.items():
            reply.write(f"\n{self._display_path(uri)}:")

            for diagnostic in diagnostics:
                location = f"[Line {self._to_range_string(diagnostic['range']['start'])}]" if 'range' in diagnostic else "[File]"
                reply.write(f"\n  {location}: {diagnostic['message']}")

        reply.write("\n</new-diagnostics>")

        return reply.getvalue()

    def go_to_definition(self, file_path: str, line: int, character: int) -> str:
        uri, version = self.notify_open(file_path)

        response = self.client.request("textDocument/definition", {
            "textDocument": {"uri": uri, "version": version},
            "position": {"line": line - 1, "character": character - 1}
        })

        if not response:
            return "No definition found"

        # The result can be a Location, a Location[], or a LocationLink[].
        locations = response if isinstance(response, list) else [response]

        lines: List[str] = []
        for location in locations:
            target_uri = location.get("targetUri") or location.get("uri")
            target_range = location.get("targetSelectionRange") or location.get("targetRange") or location.get("range")
            if not target_uri or not target_range:
                continue

            lines.append(f"Defined in {target_uri}, from {self._to_range_string(target_range['start'])} to {self._to_range_string(target_range['end'])}")

        return "\n".join(lines) or "No definition found"

    def find_references(self, file_path: str, line: int, character: int) -> str:
        uri, version = self.notify_open(file_path)

        response = self.client.request("textDocument/references", {
            "textDocument": {"uri": uri, "version": version},
            "position": {"line": line - 1, "character": character - 1},
            "context": {"includeDeclaration": True}
        })

        if not response:
            return "No references found"

        range_by_uri: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        reference_count = 0

        for item in response:
            uri = item["uri"]
            range = item["range"]

            range_by_uri[uri].append(range)
            reference_count += 1

        reply = StringIO()
        reply.write(f"Found {reference_count} references across {len(range_by_uri)} files:\n")

        for uri, ranges in range_by_uri.items():
            reply.write(f"\n{self._display_path(uri)}:\n")

            for range in ranges:
                reply.write(f"  Line {self._to_range_string(range['start'])}\n")

        return reply.getvalue()

    def _display_path(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return uri

        # Note: `str.lstrip("file://")` strips a *set of characters*, not a prefix.
        path = unquote(parsed.netloc + parsed.path)  # netloc is only non-empty for Windows-style "file://C:..." URIs
        try:
            return os.path.relpath(path)
        except ValueError:  # e.g. different drive on Windows
            return path

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

            if original_name not in arguments:
                if parameter.required:
                    return f"Error: missing required parameter '{original_name}' for tool {tool_name}"

                continue

            remapped_arguments[python_name] = arguments[original_name]

        return tool.execute(agent, **remapped_arguments)

    def collect_async_messages(self, agent: "Agent") -> List[str]:
        results: List[str] = []

        for tool in self.tool_by_name.values():
            result = tool.collect_async_message(agent)
            if result:
                results.append(result)

        return results
