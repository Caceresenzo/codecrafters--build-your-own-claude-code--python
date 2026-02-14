import argparse
import json
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME", default="anthropic/claude-haiku-4.5")  # z-ai/glm-4.5-air:free


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": args.p}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read and return the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file to read",
                            }
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    choice = chat.choices[0]

    match choice.finish_reason:
        case "tool_calls":
            for tool_call in choice.message.tool_calls:
                if tool_call.type == "function" and tool_call.function.name == "Read":
                    arguments = json.loads(tool_call.function.arguments)

                    file_path = arguments["file_path"]
                    with open(file_path, "r") as f:
                        content = f.read()

                    print(content)

        case "stop":
            print(choice.message.content)

        case _:
            raise RuntimeError(f"unexpected finish reason: {choice.finish_reason}")


if __name__ == "__main__":
    main()
