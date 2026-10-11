import json
import shlex
import sys
from typing import Any, Dict, List

from openai import OpenAI

from .skills import Skillbox
from .tools import Toolbox


class Agent:

    def __init__(
        self,
        *,
        name: str,
        client: OpenAI,
        model_name: str,
        toolbox: Toolbox,
        skillbox: Skillbox,
        include_skill_message: bool = True,
    ):
        self.name = name
        self.client = client
        self.model_name = model_name

        self.skillbox = skillbox
        self.toolbox = toolbox

        self.messages: List[Any] = []

        if include_skill_message and skillbox.skills:
            self.messages.append({
                "role": "system",
                "content": "You have access to the following skills:\n\n- " + "\n- ".join(f"{skill.name}: {skill.description}" for skill in skillbox.skills) + "\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."
            })

    def fork(
        self,
        name: str,
        include_skill_message: bool = True,
    ) -> "Agent":
        return Agent(
            name=name,
            client=self.client,
            model_name=self.model_name,
            toolbox=self.toolbox,
            skillbox=self.skillbox,
            include_skill_message=include_skill_message,
        )

    def push_user_input(self, user_input: str):
        if user_input.startswith("/"):
            arguments = shlex.split(user_input)

            skill_names: List[str] = []
            skill_arguments: List[str] = []

            for argument in arguments:
                if argument.startswith("/"):
                    skill_names.append(argument[1:])
                else:
                    skill_arguments.append(argument)

            self.use_skills(skill_names, skill_arguments)

        else:
            self.push_message({"role": "user", "content": user_input})

    def use_skills(self, names: List[str], arguments: List[str]):
        for name in names:
            skill = self.skillbox.get(name)
            if skill is None:
                print(f"{self.name}: skill: {name}: not found", file=sys.stderr)
                continue

            self.push_message({"role": "user", "content": skill.format(arguments)})

    def push_message(self, message: Dict[str, object]):
        self.messages.append(message)
        print(f"{self.name}: message: {message['role']}: {message['content']!r}", file=sys.stderr)

    def loop(self) -> str:
        while True:
            chat = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                tools=self.toolbox.tool_schemas,
            )

            if not chat.choices or len(chat.choices) == 0:
                raise RuntimeError("no choices in response")

            choice = chat.choices[0]
            message = choice.message

            self.messages.append(message)

            match choice.finish_reason:
                case "tool_calls":
                    assert message.tool_calls is not None

                    for tool_call in message.tool_calls:
                        if tool_call.type != "function":
                            raise RuntimeError(f"unexpected tool call type: {tool_call.type}")

                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)

                        arguments_str = ", ".join(f"{key}={repr(value)}" for key, value in arguments.items())
                        print(f"{self.name}: tool call: {tool_name}({arguments_str})", file=sys.stderr)

                        result = self.toolbox.use(self, tool_call.function.name, arguments)

                        message: Any = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                        self.messages.append(message)

                        print(f"{self.name}: message: {message['role']}: {message['content']!r}", file=sys.stderr)

                case "stop":
                    return message.content

                case _:
                    raise RuntimeError(f"unexpected finish reason: {choice.finish_reason}")
