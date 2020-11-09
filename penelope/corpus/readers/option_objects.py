from dataclasses import dataclass
from typing import Set


@dataclass
class AnnotationOpts:

    pos_includes: str = ''
    pos_excludes: str = "|MAD|MID|PAD|"
    passthrough_tokens: Set[str] = None
    lemmatize: bool = True
    append_pos: bool = False

    def get_pos_includes(self):
        return self.pos_includes.strip('|').split('|') if self.pos_includes is not None else None

    def get_pos_excludes(self):
        return self.pos_excludes.strip('|').split('|') if self.pos_excludes is not None else None

    def get_passthrough_tokens(self):
        if self.passthrough_tokens is None:
            return set()
        return set(self.passthrough_tokens)

    @property
    def props(self):
        return {k: v for k, v in self.__dict__.items() if k != 'props' and not k.startswith('_') and not callable(v)}
