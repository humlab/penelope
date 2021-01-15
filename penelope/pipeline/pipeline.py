from __future__ import annotations

import collections
import functools
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, List, Sequence, TypeVar, Union

from .interfaces import ContentType, DocumentPayload, ITask, PipelinePayload

if TYPE_CHECKING:
    from penelope.pipeline import CorpusConfig


_T_self = TypeVar("_T_self")
_A = TypeVar("_A")


class CorpusPipelineBase(Generic[_T_self]):
    def __init__(
        self,
        *,
        config: CorpusConfig = None,
        payload: PipelinePayload = None,
        tasks: Sequence[ITask] = None,
    ):
        self._config = config
        self._payload = payload if payload else config.pipeline_payload if config else None
        self._tasks: List[ITask] = []
        self.add(tasks or [])

    @property
    def config(self) -> CorpusConfig:
        return self._config

    @property
    def payload(self) -> PipelinePayload:
        return self._payload

    @payload.setter
    def payload(self, value: PipelinePayload):
        self._payload = value

    @property
    def tasks(self) -> List[ITask]:
        return self._tasks

    def get_prior_to(self, task: ITask) -> ITask:
        index: int = self.tasks.index(task)
        if index > 0:
            return self.tasks[index - 1]
        return None

    def get_prior_content_type(self, task: ITask) -> ContentType:
        prior_task = self.get_prior_to(task)
        if prior_task is None:
            return ContentType.NONE
        return prior_task.out_content_type

    def setup(self):
        for task in self.tasks:
            task.chain()
            task.setup()
        return self

    def resolve(self) -> Iterator[DocumentPayload]:
        """Resolves the pipeline by calling outstream on last task"""
        self.setup()
        return self.tasks[-1].outstream()

    def exhaust(self) -> _T_self:
        """Exhaust the entire pipeline, disregarding items"""
        collections.deque(self.resolve(), maxlen=0)
        return self

    def add(self, task: Union[ITask, List[ITask]]) -> _T_self:
        """Add one or more tasks to the pipeline. Hooks up a reference to pipeline for each task"""
        tasks = [task] if isinstance(task, ITask) else task
        self.tasks.extend(map(lambda x: x.hookup(self), tasks))
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def put(self, key: str, value: Any):
        self.payload.put(key, value)

    def to_list(self) -> List[DocumentPayload]:
        return [x for x in self.resolve()]

    def single(self) -> List[DocumentPayload]:
        """Resolves stream by returning the head of the tream"""
        return next(self.resolve())

    def value(self) -> Any:
        return self.single().content

    def reduce(self, function: Callable[[DocumentPayload, _A], _A], initial: _A = None) -> _A:
        """Reduces payload stream to a single value. If `initial` is `None` then first element is used."""
        if initial is None:
            return functools.reduce(function=function, sequence=self.resolve())
        return functools.reduce(function=function, sequence=self.resolve(), initial=initial)
