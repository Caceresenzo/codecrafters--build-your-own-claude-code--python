import argparse
import os

from openai import OpenAI

from .agent import Agent
from .skills import Skillbox
from .tools import BashTool, ReadTool, SkillTool, Toolbox, WriteTool

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

    skillbox = Skillbox()
    skillbox.locate()

    toolbox = Toolbox()
    toolbox.add(ReadTool())
    toolbox.add(WriteTool())
    toolbox.add(BashTool())
    toolbox.add(SkillTool(skillbox))

    claude = Agent(
        name="claude",
        client=client,
        model_name=MODEL_NAME,
        skillbox=skillbox,
        toolbox=toolbox,
    )

    claude.push_user_input(args.p)
    print(claude.loop())


if __name__ == "__main__":
    main()
