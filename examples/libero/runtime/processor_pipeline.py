from __future__ import annotations

from dataclasses import dataclass
import time
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

    def run_batch(self, contexts: Sequence[ProcessorChunkContext]) -> None:
        for context in contexts:
            self.run(context)


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

    def run_batch(self, contexts: Sequence[ProcessorChunkContext]) -> None:
        for context in contexts:
            self.run(context)


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

    def run_batch(self, contexts: Sequence[ProcessorChunkContext]) -> None:
        raw_chunks: list[ChunkExecutionRecord] = []
        task_prompts: list[str] = []
        for context in contexts:
            raw_chunk = context.raw_chunk
            if raw_chunk is None:
                raise RuntimeError("reconstruct_chunk stage must run before assemble")
            raw_chunks.append(raw_chunk)
            task_prompts.append(str(context.payload["task_prompt"]))
        assembled_chunks = self.assembler.process_chunk_batch(
            raw_chunks=raw_chunks,
            task_prompts=task_prompts,
        )
        if len(assembled_chunks) != len(contexts):
            raise RuntimeError(
                "assembler returned a mismatched chunk batch size: "
                f"got {len(assembled_chunks)} expected {len(contexts)}"
            )
        for context, assembled_chunk in zip(contexts, assembled_chunks):
            context.assembled_chunk = assembled_chunk


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
        assembled_chunks = self.process_payload_batch(
            payloads=(payload,),
            timer=timer,
        )
        if len(assembled_chunks) != 1:
            raise RuntimeError(
                "processor pipeline expected exactly one assembled chunk, "
                f"got {len(assembled_chunks)}"
            )
        return assembled_chunks[0]

    def process_payload_batch(
        self,
        payloads: Sequence[dict[str, Any]],
        *,
        timer: Any | None = None,
    ) -> list[AssemblyResult]:
        contexts = [ProcessorChunkContext(payload=dict(payload)) for payload in payloads]
        if not contexts:
            return []

        for stage in self._chunk_stages:
            stage_name = str(stage.name)
            self._run_stage_batch(
                stage_name=stage_name,
                timer=timer,
                batch_size=len(contexts),
                fn=lambda current_stage=stage: self._run_chunk_stage_batch(
                    current_stage,
                    contexts,
                ),
            )

        assembled_chunks: list[AssemblyResult] = []
        for context in contexts:
            assembled_chunk = context.assembled_chunk
            if assembled_chunk is None:
                raise RuntimeError("processor pipeline did not assemble transitions")
            assembled_chunks.append(assembled_chunk)

        for stage in self._transition_stages:
            stage_name = str(stage.name)

            def _run_transition_stage_batch(
                current_stage: ProcessorTransitionStage = stage,
            ) -> None:
                nonlocal assembled_chunks
                assembled_chunks = self._run_transition_stage_batch(
                    current_stage,
                    assembled_chunks,
                    contexts,
                )

            self._run_stage_batch(
                stage_name=stage_name,
                timer=timer,
                batch_size=len(contexts),
                fn=_run_transition_stage_batch,
            )

        return assembled_chunks

    @staticmethod
    def _run_stage(*, stage_name: str, timer: Any | None, fn: Any) -> None:
        if timer is None:
            fn()
            return
        with timer.context(str(stage_name)):
            fn()

    @staticmethod
    def _run_stage_batch(
        *,
        stage_name: str,
        timer: Any | None,
        batch_size: int,
        fn: Any,
    ) -> None:
        if timer is None:
            fn()
            return
        start_time = time.time()
        fn()
        duration = time.time() - start_time
        RolloutProcessorPipeline._record_timer_duration(
            timer=timer,
            stage_name=stage_name,
            duration_s=duration,
            count=max(1, int(batch_size)),
        )

    @staticmethod
    def _run_chunk_stage_batch(
        stage: ProcessorChunkStage,
        contexts: Sequence[ProcessorChunkContext],
    ) -> None:
        run_batch = getattr(stage, "run_batch", None)
        if callable(run_batch):
            run_batch(contexts)
            return
        for context in contexts:
            stage.run(context)

    @staticmethod
    def _run_transition_stage_batch(
        stage: ProcessorTransitionStage,
        assembled_chunks: Sequence[AssemblyResult],
        contexts: Sequence[ProcessorChunkContext],
    ) -> list[AssemblyResult]:
        run_batch = getattr(stage, "run_batch", None)
        if callable(run_batch):
            return list(run_batch(assembled_chunks, contexts=contexts))

        next_chunks: list[AssemblyResult] = []
        for assembled_chunk, context in zip(assembled_chunks, contexts):
            next_chunks.append(stage.run(assembled_chunk, context=context))
        return next_chunks

    @staticmethod
    def _record_timer_duration(
        *,
        timer: Any,
        stage_name: str,
        duration_s: float,
        count: int,
    ) -> None:
        timer.times[str(stage_name)] += float(duration_s)
        timer.counts[str(stage_name)] += max(1, int(count))


__all__ = [
    "AssembleTransitionsStage",
    "NormalizeChunkStage",
    "ProcessorChunkContext",
    "ProcessorChunkStage",
    "ProcessorTransitionStage",
    "ReconstructChunkExecutionRecordStage",
    "RolloutProcessorPipeline",
]
