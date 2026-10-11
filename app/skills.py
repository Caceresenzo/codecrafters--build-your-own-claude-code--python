from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import List


@dataclass
class Skill:
    path: Path
    name: str
    description: str


class Skillbox:

    def __init__(self):
        self.skills: List[Skill] = []

    def add(self, skill: Skill):
        self.skills.append(skill)

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

                self.add(Skill(
                    path=markdown_path,
                    name=name,
                    description=description
                ))
