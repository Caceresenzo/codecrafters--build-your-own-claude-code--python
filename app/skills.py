import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List, Optional


@dataclass
class Skill:
    directory: Path
    name: str
    description: str
    content: str

    def format(self, arguments: list[str]) -> str:
        formatted = self.content

        if "$ARGUMENTS" in formatted:
            formatted = formatted.replace("$ARGUMENTS", " ".join(arguments))

        def replacer(match: re.Match[str]):
            index = int(match.group(1) or match.group(2))
            return arguments[index] if index < len(arguments) else ""

        formatted = re.sub(rf"(?:\$(\d+)|\$ARGUMENTS\[(\d+)\])", replacer, formatted)

        return formatted


class Skillbox:

    def __init__(self):
        self.skills: List[Skill] = []

    def add(self, skill: Skill):
        self.skills.append(skill)

    def get(self, name: str) -> Optional[Skill]:
        for skill in self.skills:
            if skill.name == name:
                return skill

    def locate(self):
        markdown_files = glob(".claude/skills/*/SKILL.md")

        for markdown_file in markdown_files:
            markdown_path = Path(markdown_file)

            with markdown_path.open("r") as fd:
                if fd.readline().strip() != "---":
                    continue

                name: str = ""
                description: str = ""

                while True:
                    line = fd.readline().strip()
                    if line == "---":
                        break

                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    if key == "name":
                        name = value
                    elif key == "description":
                        description = value

                content = fd.read().strip()

                self.add(Skill(
                    directory=markdown_path.parent,
                    name=name,
                    description=description,
                    content=content
                ))
