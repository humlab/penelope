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

    # FIXME; #16 Serialization of annotatioon opts fails
    @property
    def props(self):
        return dict(
            pos_includes=self.pos_includes,
            pos_excludes=self.pos_excludes,
            passthrough_tokens=list(self.passthrough_tokens),
            lemmatize=self.lemmatize,
            append_pos=self.append_pos
        )
