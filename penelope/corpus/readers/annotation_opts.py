from dataclasses import dataclass


@dataclass
class AnnotationOpts:

    pos_includes: str = ''
    pos_excludes: str = "|MAD|MID|PAD|"
    lemmatize: bool = True
    append_pos: bool = False

    def get_pos_includes(self):
        return self.pos_includes.strip('|').split('|') if self.pos_includes is not None else None

    def get_pos_excludes(self):
        return self.pos_excludes.strip('|').split('|') if self.pos_excludes is not None else None

    @property
    def props(self):
        return {k: v for k, v in self.__dict__.items() if k != 'props' and not k.startswith('_') and not callable(v)}
