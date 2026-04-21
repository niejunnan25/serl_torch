from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import Sequence

from .processor_protocol import NormalizedChunkResult
from .processor_protocol import normalize_chunk_result
from .processor_protocol import reconstruct_chunk_execution_record_from_normalized
from .transition_assembly import AssemblyResult
from .transition_assembly import BatchAwareLiberoTransitionAssembler
from .transition_assembly import ChunkExecutionRecord


@dataclass
class ProcessorChunkContext:
    payload: dict[str, Any]
    normalized_chunk: NormalizedChunkResult | None = None
    raw_chunk: ChunkExecutionRecord | None = None
    assembled_chunk: AssemblyResult | None = None


class ProcessorChunkStage(Protocol):
    name: str

    def run(self, context: ProcessorChunkContext) -> None: ...


class ProcessorTransitionStage(Protocol):
    name: str

    def run(
        self,
        assembled_chunk: AssemblyResult,
        *,
        context: ProcessorChunkContext,
    ) -> AssemblyResult: ...


@dataclass(frozen=True, slots=True)
class NormalizeChunkStage:
    name: str = "normalize_chunk"

    def run(self, context: ProcessorChunkContext) -> None:
        context.normalized_chunk = normalize_chunk_result(
            dict(context.payload["chunk_result"])
        )


@dataclass(frozen=True, slots=True)
class ReconstructChunkExecutionRecordStage:
    assembler: BatchAwareLiberoTransitionAssembler
    name: str = "reconstruct_chunk"

    def run(self, context: ProcessorChunkContext) -> None:
        normalized_chunk = context.normalized_chunk
        if normalized_chunk is None:
            raise RuntimeError("normalize_chunk stage must run before reconstruct")
        context.raw_chunk = reconstruct_chunk_execution_record_from_normalized(
            payload=context.payload,
            normalized_chunk=normalized_chunk,
            assembler=self.assembler,
        )


@dataclass(frozen=True, slots=True)
class AssembleTransitionsStage:
    assembler: BatchAwareLiberoTransitionAssembler
    name: str = "assemble_transitions"

    def run(self, context: ProcessorChunkContext) -> None:
        raw_chunk = context.raw_chunk
        if raw_chunk is None:
            raise RuntimeError("reconstruct_chunk stage must run before assemble")
        context.assembled_chunk = self.assembler.process_chunk(
            raw=raw_chunk,
            task_prompt=str(context.payload["task_prompt"]),
        )


class RolloutProcessorPipeline:
    def __init__(
        self,
        *,
        chunk_stages: Sequence[ProcessorChunkStage],
        transition_stages: Sequence[ProcessorTransitionStage] = (),
    ) -> None:
        self._chunk_stages = tuple(chunk_stages)
        self._transition_stages = tuple(transition_stages)

    @classmethod
    def for_libero(
        cls,
        *,
        assembler: BatchAwareLiberoTransitionAssembler,
        transition_stages: Sequence[ProcessorTransitionStage] = (),
    ) -> "RolloutProcessorPipeline":
        return cls(
            chunk_stages=(
                NormalizeChunkStage(),
                ReconstructChunkExecutionRecordStage(assembler=assembler),
                AssembleTransitionsStage(assembler=assembler),
            ),
            transition_stages=transition_stages,
        )

    def process_payload(
        self,
        payload: dict[str, Any],
        *,
        timer: Any | None = None,
    ) -> AssemblyResult:
        context = ProcessorChunkContext(payload=dict(payload))

        for stage in self._chunk_stages:
            self._run_stage(stage_name=stage.name, timer=timer, fn=lambda: stage.run(context))

        assembled_chunk = context.assembled_chunk
        if assembled_chunk is None:
            raise RuntimeError("processor pipeline did not assemble transitions")

        for stage in self._transition_stages:
            stage_name = str(stage.name)

            def _run_transition_stage() -> None:
                nonlocal assembled_chunk
                assembled_chunk = stage.run(assembled_chunk, context=context)

            self._run_stage(stage_name=stage_name, timer=timer, fn=_run_transition_stage)

        return assembled_chunk

    @staticmethod
    def _run_stage(*, stage_name: str, timer: Any | None, fn: Any) -> None:
        if timer is None:
            fn()
            return
        with timer.context(str(stage_name)):
            fn()


__all__ = [
    "AssembleTransitionsStage",
    "NormalizeChunkStage",
    "ProcessorChunkContext",
    "ProcessorChunkStage",
    "ProcessorTransitionStage",
    "ReconstructChunkExecutionRecordStage",
    "RolloutProcessorPipeline",
]
