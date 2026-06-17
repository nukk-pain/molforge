from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class VinaNormalizer(Protocol):
    def normalize(self, scores: list[float]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class AbsoluteVinaNormalizer:
    min_score: float = -12.0
    max_score: float = 0.0

    def normalize(self, scores: list[float]) -> list[float]:
        return [self.normalize_score(score) for score in scores]

    def normalize_score(self, score: float) -> float:
        clamped = min(self.max_score, max(self.min_score, float(score)))
        span = self.max_score - self.min_score
        if span <= 0:
            raise ValueError("AbsoluteVinaNormalizer requires max_score > min_score.")
        return (self.max_score - clamped) / span


@dataclass(frozen=True, slots=True)
class PercentileVinaNormalizer:
    def normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        if len(scores) == 1:
            return [1.0]

        normalized = [0.0] * len(scores)
        indexed_scores = sorted(enumerate(scores), key=lambda item: float(item[1]))
        denominator = len(scores) - 1

        group_start = 0
        while group_start < len(indexed_scores):
            group_end = group_start + 1
            current_score = float(indexed_scores[group_start][1])
            while group_end < len(indexed_scores):
                next_score = float(indexed_scores[group_end][1])
                if next_score != current_score:
                    break
                group_end += 1

            average_rank = (group_start + group_end - 1) / 2.0
            percentile = 1.0 - (average_rank / denominator)
            for original_index, _score in indexed_scores[group_start:group_end]:
                normalized[original_index] = percentile

            group_start = group_end

        return normalized


@dataclass(frozen=True, slots=True)
class AdaptiveVinaNormalizer:
    min_candidates: int = 5
    absolute: AbsoluteVinaNormalizer = AbsoluteVinaNormalizer()
    percentile: PercentileVinaNormalizer = PercentileVinaNormalizer()

    def normalize(self, scores: list[float]) -> list[float]:
        if len(scores) < self.min_candidates:
            return self.absolute.normalize(scores)
        return self.percentile.normalize(scores)
