"""Frozen metadata-only expanding splits; no X, Y, models or scoring."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.download import sha256
from src.validation.time_split import read_header


@dataclass(frozen=True)
class ValidationProtocol:
    development_start: int
    development_end: int
    research_start: int
    research_end: int
    initial_train_size: int
    validation_window: int
    step_size: int
    purge_size: int
    embargo_size: int
    fold_count: int
    internal_holdout_start: int
    internal_holdout_end: int
    final_competition_start: int
    final_competition_end: int

    @classmethod
    def from_mapping(cls, config: dict) -> "ValidationProtocol":
        result = cls(**{f.name: config[f.name] for f in fields(cls)})
        if config.get("shuffle") is not False or config.get("method") != "expanding_window":
            raise ValueError("Only unshuffled expanding windows are allowed.")
        if any(type(getattr(result, f.name)) is not int for f in fields(cls)):
            raise ValueError("Validation boundaries and lengths must be integers.")
        if (result.research_start != result.development_start
                or result.research_end + 1 != result.internal_holdout_start
                or result.internal_holdout_end != result.development_end
                or result.development_end + 1 != result.final_competition_start
                or result.final_competition_end < result.final_competition_start):
            raise ValueError("Research, holdout and competition must be ordered and disjoint.")
        if (min(result.initial_train_size, result.validation_window, result.step_size, result.fold_count) < 1
                or min(result.purge_size, result.embargo_size) < 0
                or result.step_size < result.validation_window + result.embargo_size):
            raise ValueError("Invalid windows, purge or post-validation embargo.")
        return result


@dataclass(frozen=True)
class WalkForwardFold:
    number: int
    train_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    purged_ids: tuple[int, ...]
    embargo_ids: tuple[int, ...]

    def metadata(self) -> dict:
        return {"fold": self.number, "train_start": self.train_ids[0], "train_end": self.train_ids[-1],
            "training_rows": len(self.train_ids), "validation_start": self.validation_ids[0],
            "validation_end": self.validation_ids[-1], "validation_rows": len(self.validation_ids),
            "purged_ids": list(self.purged_ids), "embargo_ids": list(self.embargo_ids)}


def load_protocol(path: Path) -> ValidationProtocol:
    mapping = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (mapping.get("internal_holdout_locked") is not True or mapping.get("final_competition_locked") is not True
            or (mapping["development_start"], mapping["development_end"], mapping["final_competition_start"],
                mapping["final_competition_end"]) != (0, 8867, 8868, 9047)):
        raise ValueError("The fixed outer reservation must remain locked.")
    result = ValidationProtocol.from_mapping(mapping)
    make_folds(np.arange(result.development_start, result.development_end + 1), result)
    return result


def make_folds(date_ids, protocol: ValidationProtocol) -> tuple[WalkForwardFold, ...]:
    """Use IDs only; prior validation may become later training once historical.

    Nonzero embargo means post-validation rows never enter subsequent training,
    and step_size must also skip them in later validation. Default embargo=0.
    Purge removes rows immediately preceding each validation decision boundary.
    """
    dates = np.asarray(date_ids)
    expected = np.arange(protocol.development_start, protocol.development_end + 1)
    if dates.ndim != 1 or not np.issubdtype(dates.dtype, np.integer) or not np.array_equal(dates, expected):
        raise ValueError("IDs must be the ordered, unique development range; never sort or shuffle.")
    first = protocol.research_start + protocol.initial_train_size + protocol.purge_size
    folds, excluded = [], set()
    for start in range(first, protocol.research_end + 1, protocol.step_size):
        stop = start + protocol.validation_window
        if stop - 1 > protocol.research_end:
            raise ValueError("Incomplete validation tail; choose explicit full windows.")
        train = tuple(i for i in range(protocol.research_start, start - protocol.purge_size) if i not in excluded)
        if not train:
            raise ValueError("Empty historical training window.")
        embargo = tuple(range(stop, min(stop + protocol.embargo_size, protocol.research_end + 1)))
        folds.append(WalkForwardFold(len(folds) + 1, train, tuple(range(start, stop)),
                                    tuple(range(start - protocol.purge_size, start)), embargo))
        excluded.update(embargo)
    if len(folds) != protocol.fold_count:
        raise ValueError("Fold count differs from frozen protocol.")
    return tuple(folds)


def holdout_training_ids(protocol: ValidationProtocol) -> tuple[int, ...]:
    """Frozen final research refit stops before the holdout's one-row purge."""
    folds = make_folds(np.arange(protocol.development_start, protocol.development_end + 1), protocol)
    excluded = {i for fold in folds for i in fold.embargo_ids}
    return tuple(i for i in range(protocol.research_start, protocol.internal_holdout_start - protocol.purge_size)
                 if i not in excluded)


def load_research_prefix(train_path: Path, split_plan: dict, protocol: ValidationProtocol) -> pd.DataFrame:
    """Future research loader: never parse internal holdout or final values.

    Unlike historical Phase 2 EDA, the future research boundary is 7859.
    This helper is exercised only on artificial inputs in Phase 2.5.
    """
    if (protocol.final_competition_start != 8868 or protocol.development_end != 8867
            or split_plan["competition_simulation_locked"] is not True
            or sha256(train_path) != split_plan["source_train_sha256"]
            or read_header(train_path) != split_plan["train_columns"]):
        raise ValueError("Research source/schema or outer reservation differs.")
    frame = pd.read_csv(train_path, nrows=protocol.internal_holdout_start)
    if not np.array_equal(frame["date_id"], np.arange(protocol.internal_holdout_start)):
        raise ValueError("Invalid research prefix chronology.")
    return frame
