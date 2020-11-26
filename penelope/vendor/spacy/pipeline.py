from typing import Any, Iterator, List, Sequence, Union

from . import interfaces, tasks_mixin


class CorpusPipeline(tasks_mixin.PipelineShortcutMixIn):
    def __init__(
        self,
        *,
        payload: interfaces.PipelinePayload,
        tasks: Sequence[interfaces.ITask] = None,
    ):
        self._payload = payload
        self._tasks: List[interfaces.ITask] = []
        self.add(tasks or [])
        self.resolved = False

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

    def resolve(self) -> Iterator[interfaces.DocumentPayload]:
        """Resolves the pipeline by calling outstream on last task"""
        if not self.resolved:
            for task in self.tasks:
                task.chain()
                task.setup()
            self.resolved = True
        return self.tasks[-1].outstream()

    def add(self, task: Union[interfaces.ITask, List[interfaces.ITask]]) -> "CorpusPipeline":
        """Add one or more tasks to the pipeline. Hooks up a reference to pipeline for each task"""
        tasks = [task] if isinstance(task, interfaces.ITask) else task
        self.tasks.extend(map(lambda x: x.hookup(self), tasks))
        return self

    def get(self, key: str, default=None):
        return self.payload.get(key, default)

    def put(self, key: str, value: Any):
        self.payload.put[key] = value


class SpacyPipeline(CorpusPipeline):
    pass
