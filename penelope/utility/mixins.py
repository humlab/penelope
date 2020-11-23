class PropsMixIn:
    @property
    def props(self):
        return {k: v for k, v in self.__dict__.items() if k != 'props' and not k.startswith('_') and not callable(v)}
