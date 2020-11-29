import collections
from typing import Any, Iterator, List, Sequence, Union

from penelope.pipeline.interfaces import ContentType

from . import interfaces, tasks_mixin


# FIXME: Move SpacyMixIn to SpacyPipeline (gives an error)
class CorpusPipeline(tasks_mixin.PipelineShortcutMixIn, tasks_mixin.SpacyPipelineShortcutMixIn):
    def __init__(
        self,
        *,
        payload: interfaces.PipelinePayload,
        tasks: Sequence[interfaces.ITask] = None,
    ):
        self._payload = payload
        self._tasks: List[interfaces.ITask] = []
        self.add(tasks or [])
        # self.resolved = False

    @property
    def payload(self) -> interfaces.PipelinePayload:
        return self._payload

    @property
    def tasks(self) -> List[interfaces.ITask]:
        return self._tasks

    def get_prior_to(self, task: interfaces.ITask) -> interfaces.ITask:
        index: int = self.tasks.index(task)
        if index > 0:
            return self.tasks[index - 1]
        return None

    def get_prior_content_type(self, task: interfaces.ITask) -> ContentType:
        prior_task = self.get_prior_to(task)
        if prior_task is None:
            return ContentType.NONE
        return prior_task.out_content_type

    def setup(self):
        for task in self.tasks:
            task.chain()
            task.setup()
        return self

    def resolve(self) -> Iterator[interfaces.DocumentPayload]:
        """Resolves the pipeline by calling outstream on last task"""
        # if not self.resolved:
        self.setup()
        #    self.resolved = True
        return self.tasks[-1].outstream()

    def exhaust(self) -> "CorpusPipeline":
        # if self.resolved:
        #     raise interfaces.PipelineError("cannot exhaust an already resolved pipeline")
        collections.deque(self.resolve(), maxlen=0)
        return self

    def add(self, task: Union[interfaces.ITask, List[interfaces.ITask]]) -> "CorpusPipeline":
        """Add one or more tasks to the pipeline. Hooks up a reference to pipeline for each task"""
        tasks = [task] if isinstance(task, interfaces.ITask) else task
        self.tasks.extend(map(lambda x: x.hookup(self), tasks))
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def put(self, key: str, value: Any):
        self.payload.put(key, value)


class SpacyPipeline(CorpusPipeline, tasks_mixin.SpacyPipelineShortcutMixIn):
    def nlp(self):
        self.get('spacy_nlp', None)
