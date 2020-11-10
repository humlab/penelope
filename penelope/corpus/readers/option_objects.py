from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class AnnotationOpts:

    pos_includes: str = ''
    pos_excludes: str = "|MAD|MID|PAD|"
    passthrough_tokens: List[str] = field(default_factory=list)
    lemmatize: bool = True
    append_pos: bool = False

    def get_pos_includes(self):
        return self.pos_includes.strip('|').split('|') if self.pos_includes is not None else None

    def get_pos_excludes(self):
        return self.pos_excludes.strip('|').split('|') if self.pos_excludes is not None else None

    def get_passthrough_tokens(self) -> Set[str]:
        if self.passthrough_tokens is None:
            return set()
        return set(self.passthrough_tokens)

    @property
    def props(self):
        return dict(
            pos_includes=self.pos_includes,
            pos_excludes=self.pos_excludes,
            passthrough_tokens=(self.passthrough_tokens or []),
            lemmatize=self.lemmatize,
            append_pos=self.append_pos,
        )
