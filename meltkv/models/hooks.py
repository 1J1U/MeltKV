from dataclasses import dataclass

from ..inject import ContextList


@dataclass(frozen=True)
class ModelHooks:
    init_context: ContextList
    baseline_context: ContextList
    evaluation_context: ContextList
