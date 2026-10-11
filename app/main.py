import argparse
import json
import os
import shlex
import sys
from typing import Any, List

from openai import OpenAI

from .skills import Skill, Skillbox
from .tools import BashTool, ReadTool, Toolbox, WriteTool

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", default="anthropic/claude-haiku-4.5")  # nvidia/nemotron-3.5-lightning:free


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", required=True)
    args = parser.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    toolbox = Toolbox()
    toolbox.add(ReadTool)
    toolbox.add(WriteTool)
    toolbox.add(BashTool)

    skillbox = Skillbox()
    skillbox.locate()

    skill_messages: List[Any] = (
        [{"role": "system", "content": "You have access to the following skills:\n\n- " + "\n- ".join(f"{skill.name}: {skill.description}" for skill in skillbox.skills)}]
        if skillbox.skills
        else []
    )

    user_input = args.p

    messages: List[Any] = [
        *skill_messages,
    ]

    if user_input.startswith("/"):
        arguments = shlex.split(user_input)

        skill_arguments: List[str] = []
        skills: List[Skill] = []

        for argument in arguments:
            if argument.startswith("/"):
                skill_name = argument[1:]

                skill = skillbox.get(skill_name)
                if skill is None:
                    print(f"skill: {skill_name}: not found", file=sys.stderr)
                else:
                    skills.append(skill)

            else:
                skill_arguments.append(argument)

        for skill in skills:
            formatted = skill.format(skill_arguments)

            prefix = f"User invoked a the skill: {skill.name} (located at {skill.directory})"
            messages.append({"role": "user", "content": f"{prefix}\n{formatted}"})

    else:
        messages.append({"role": "user", "content": user_input})

    for message in messages:
        print(f"message: {message['role']}: {message['content']!r}", file=sys.stderr)

    while True:
        chat = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=toolbox.tool_schemas,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        choice = chat.choices[0]
        message = choice.message

        messages.append(message)

        match choice.finish_reason:
            case "tool_calls":
                assert message.tool_calls is not None

                for tool_call in message.tool_calls:
                    if tool_call.type != "function":
                        raise RuntimeError(f"unexpected tool call type: {tool_call.type}")

                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)

                    arguments_str = ", ".join(f"{key}={repr(value)}" for key, value in arguments.items())
                    print(f"tool call: {tool_name}({arguments_str})", file=sys.stderr)

                    result = toolbox.use(tool_call.function.name, arguments)

                    message: Any = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                    messages.append(message)

                    print(f"message: {message['role']}: {message['content']!r}", file=sys.stderr)

            case "stop":
                print(message.content)
                break

            case _:
                raise RuntimeError(f"unexpected finish reason: {choice.finish_reason}")


if __name__ == "__main__":
    main()
