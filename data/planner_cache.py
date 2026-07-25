"""Planner-cache 契约、索引和训练前校验。

该模块不依赖 PyTorch、LeRobot 或 Qwen API。它是 M1 训练数据与未来在线
Planner 共用的稳定边界：训练侧只能通过 ``episode_index + frame_index`` 查询
一个已经审核通过的子任务，不能在 cache 缺失时悄悄退回整任务 instruction。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from planner.contract import (
    ACTION_WHITELIST,
    PLANNER_SEMANTIC_CONTRACT_VERSION,
    POSE_ADJUST_ACTION,
    PlannerContractError,
    derive_code_policy,
    instruction_contract_text,
    normalize_planner_subtask,
)


CACHE_SCHEMA_VERSION = "vqap_subtask_cache_v2"
INSTRUCTION_CONTRACT_VERSION = PLANNER_SEMANTIC_CONTRACT_VERSION
ACTION_SPACE_DELTA = "delta"
TRAIN_CACHE_ROLE = "train_lerobot"
RAW_VALIDATION_CACHE_ROLE = "raw_validation"
CacheRole = Literal["train_lerobot", "raw_validation"]
VALID_CACHE_ROLES = frozenset({TRAIN_CACHE_ROLE, RAW_VALIDATION_CACHE_ROLE})

IssueLevel = Literal["error", "warning"]


class PlannerCacheError(ValueError):
    """planner-cache 格式或查询不满足 M1 契约时抛出。"""


@dataclass(frozen=True)
class ValidationIssue:
    """一条可定位到 episode 的契约检查结果。"""

    level: IssueLevel
    code: str
    message: str
    episode_key: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "episode_key": self.episode_key,
        }


@dataclass
class PlannerCacheValidationReport:
    """cache 与 conversion progress 的完整校验报告。"""

    cache_path: str | None = None
    conversion_progress_path: str | None = None
    cache_role: str | None = None
    checked_episodes: int = 0
    raw_geometry_checked: bool = False
    lerobot_geometry_checked: bool = False
    conversion_checked: bool = False
    training_eligible: bool = False
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        level: IssueLevel,
        code: str,
        message: str,
        *,
        episode_key: str | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(level, code, message, episode_key))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.level == "warning"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_path": self.cache_path,
            "conversion_progress_path": self.conversion_progress_path,
            "cache_role": self.cache_role,
            "checked_episodes": self.checked_episodes,
            "raw_geometry_checked": self.raw_geometry_checked,
            "lerobot_geometry_checked": self.lerobot_geometry_checked,
            "conversion_checked": self.conversion_checked,
            "training_eligible": self.training_eligible,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "valid": self.is_valid,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class PlannerCacheValidationError(PlannerCacheError):
    """严格加载时，用完整报告解释 cache 为什么不可用于训练。"""

    def __init__(self, report: PlannerCacheValidationReport) -> None:
        self.report = report
        preview = "; ".join(
            f"{issue.code}: {issue.message}" for issue in report.errors[:3]
        )
        super().__init__(
            f"Planner cache validation failed with {len(report.errors)} error(s)"
            + (f": {preview}" if preview else "")
        )


@dataclass(frozen=True)
class PlannerSegment:
    """一个可直接用于训练或推理的子任务段。"""

    segment_index: int
    action: str
    instruction: str
    raw_start_frame: int
    raw_end_frame: int
    lerobot_start_frame: int | None
    lerobot_end_frame: int | None
    code_policy: str
    stored_code_policy: str | None
    confidence: float
    boundary_reason: str

    def contains_lerobot_frame(self, frame_index: int) -> bool:
        if self.lerobot_start_frame is None or self.lerobot_end_frame is None:
            return False
        return self.lerobot_start_frame <= frame_index <= self.lerobot_end_frame


@dataclass(frozen=True)
class PlannerEpisode:
    """One episode record in either LeRobot-indexed or raw-validation form."""

    cache_key: str
    output_episode_index: int | None
    raw_episode_key: str
    task_name: str
    variation: str
    episode: str
    task_instruction: str
    all_frames: int
    lerobot_frames: int | None
    needs_review: bool
    segments: tuple[PlannerSegment, ...]


@dataclass(frozen=True)
class SubtaskContext:
    """模型侧所需的最小 planner 上下文。"""

    episode_index: int
    frame_index: int
    raw_episode_key: str
    segment_index: int
    action: str
    instruction: str
    code_policy: str
    lerobot_start_frame: int
    lerobot_end_frame: int
    cache_fingerprint: str


def _as_int(value: Any, field_name: str, *, episode_key: str) -> int:
    if isinstance(value, bool):
        raise PlannerCacheError(f"{episode_key}.{field_name} must be an integer, got bool.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PlannerCacheError(f"{episode_key}.{field_name} must be an integer, got {value!r}.") from exc
    return parsed


def _as_optional_int(value: Any, field_name: str, *, episode_key: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field_name, episode_key=episode_key)


def _as_nonempty_string(value: Any, field_name: str, *, episode_key: str) -> str:
    parsed = str(value or "").strip()
    if not parsed:
        raise PlannerCacheError(f"{episode_key}.{field_name} must be a non-empty string.")
    return parsed


def _load_segment(segment: Any, *, episode_key: str) -> PlannerSegment:
    if not isinstance(segment, dict):
        raise PlannerCacheError(f"{episode_key}.segments entries must be objects.")
    try:
        subtask = normalize_planner_subtask(segment)
    except PlannerContractError as exc:
        raise PlannerCacheError(f"{episode_key}.segments has invalid semantics: {exc}") from exc
    code_policy = derive_code_policy(subtask.action)
    stored_code_policy_value = segment.get("code_policy")
    stored_code_policy = (
        None if stored_code_policy_value is None else str(stored_code_policy_value).strip()
    )
    try:
        confidence = float(segment.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise PlannerCacheError(f"{episode_key}.segments.confidence must be numeric.") from exc
    if not 0.0 <= confidence <= 1.0:
        raise PlannerCacheError(
            f"{episode_key}.segments.confidence must be in [0, 1], got {confidence}."
        )
    return PlannerSegment(
        segment_index=_as_int(segment.get("segment_index"), "segments.segment_index", episode_key=episode_key),
        action=subtask.action,
        instruction=subtask.instruction,
        raw_start_frame=_as_int(segment.get("raw_start_frame"), "segments.raw_start_frame", episode_key=episode_key),
        raw_end_frame=_as_int(segment.get("raw_end_frame"), "segments.raw_end_frame", episode_key=episode_key),
        lerobot_start_frame=_as_optional_int(
            segment.get("lerobot_start_frame"), "segments.lerobot_start_frame", episode_key=episode_key
        ),
        lerobot_end_frame=_as_optional_int(
            segment.get("lerobot_end_frame"), "segments.lerobot_end_frame", episode_key=episode_key
        ),
        code_policy=code_policy,
        stored_code_policy=stored_code_policy,
        confidence=confidence,
        boundary_reason=str(segment.get("boundary_reason", "")).strip(),
    )


def _load_episode(cache_key: str, record: Any) -> PlannerEpisode:
    if not isinstance(record, dict):
        raise PlannerCacheError(f"{cache_key} record must be an object.")
    output_episode_index = _as_optional_int(
        record.get("output_episode_index"), "output_episode_index", episode_key=cache_key
    )
    segments_value = record.get("segments")
    if not isinstance(segments_value, list) or not segments_value:
        raise PlannerCacheError(f"{cache_key}.segments must be a non-empty list.")
    return PlannerEpisode(
        cache_key=cache_key,
        output_episode_index=output_episode_index,
        raw_episode_key=_as_nonempty_string(record.get("raw_episode_key"), "raw_episode_key", episode_key=cache_key),
        task_name=_as_nonempty_string(record.get("task_name"), "task_name", episode_key=cache_key),
        variation=_as_nonempty_string(record.get("variation"), "variation", episode_key=cache_key),
        episode=_as_nonempty_string(record.get("episode"), "episode", episode_key=cache_key),
        task_instruction=_as_nonempty_string(
            record.get("task_instruction"), "task_instruction", episode_key=cache_key
        ),
        all_frames=_as_int(record.get("all_frames"), "all_frames", episode_key=cache_key),
        lerobot_frames=_as_optional_int(
            record.get("lerobot_frames"), "lerobot_frames", episode_key=cache_key
        ),
        needs_review=bool(record.get("needs_review", False)),
        segments=tuple(_load_segment(segment, episode_key=cache_key) for segment in segments_value),
    )


def _validate_episode_geometry(
    episode: PlannerEpisode,
    report: PlannerCacheValidationReport,
    *,
    cache_role: CacheRole,
) -> None:
    key = episode.cache_key
    if episode.all_frames < 1:
        report.add("error", "invalid_raw_frames", "all_frames must be >= 1.", episode_key=key)
        return
    if cache_role == TRAIN_CACHE_ROLE:
        if episode.output_episode_index is None or episode.output_episode_index < 0:
            report.add(
                "error",
                "invalid_episode_index",
                "train_lerobot requires output_episode_index >= 0.",
                episode_key=key,
            )
        elif str(episode.output_episode_index) != key:
            report.add(
                "error",
                "cache_key_mismatch",
                f"Cache key must equal output_episode_index ({episode.output_episode_index}).",
                episode_key=key,
            )
        if episode.lerobot_frames is None or episode.lerobot_frames < 1:
            report.add(
                "error",
                "invalid_lerobot_frames",
                "train_lerobot requires lerobot_frames >= 1.",
                episode_key=key,
            )
    else:
        if episode.output_episode_index is not None:
            report.add(
                "error",
                "unexpected_episode_index",
                "raw_validation requires output_episode_index=null.",
                episode_key=key,
            )
        if key != episode.raw_episode_key:
            report.add(
                "error",
                "cache_key_mismatch",
                "raw_validation cache key must equal raw_episode_key.",
                episode_key=key,
            )
    if episode.needs_review:
        report.add(
            "error",
            "needs_review",
            "Episode is marked needs_review and cannot enter M1 training.",
            episode_key=key,
        )

    previous_lerobot_end = -1
    previous_raw_end = -1
    for expected_index, segment in enumerate(episode.segments):
        if segment.stored_code_policy not in (None, segment.code_policy):
            report.add(
                "warning",
                "planner_code_policy_ignored",
                f"Stored code_policy={segment.stored_code_policy!r} was ignored; "
                f"derived {segment.code_policy!r} from action={segment.action!r}.",
                episode_key=key,
            )
        if segment.segment_index != expected_index:
            report.add(
                "error",
                "segment_index_mismatch",
                f"Expected segment_index={expected_index}, got {segment.segment_index}.",
                episode_key=key,
            )
        if segment.raw_start_frame < 0 or segment.raw_end_frame < segment.raw_start_frame:
            report.add("error", "invalid_raw_range", "Invalid raw frame range.", episode_key=key)
        if segment.raw_start_frame != previous_raw_end + 1:
            report.add(
                "error",
                "raw_gap_or_overlap",
                f"Expected start={previous_raw_end + 1}, got {segment.raw_start_frame}.",
                episode_key=key,
            )
        previous_raw_end = segment.raw_end_frame
        if cache_role == TRAIN_CACHE_ROLE:
            if segment.lerobot_start_frame is None or segment.lerobot_end_frame is None:
                report.add(
                    "error",
                    "missing_lerobot_range",
                    "train_lerobot segments require LeRobot frame ranges.",
                    episode_key=key,
                )
                continue
            if (
                segment.lerobot_start_frame < 0
                or segment.lerobot_end_frame < segment.lerobot_start_frame
            ):
                report.add(
                    "error", "invalid_lerobot_range", "Invalid LeRobot frame range.", episode_key=key
                )
            if (
                episode.lerobot_frames is not None
                and segment.lerobot_end_frame >= episode.lerobot_frames
            ):
                report.add(
                    "error",
                    "lerobot_range_out_of_bounds",
                    f"Segment ends at {segment.lerobot_end_frame}, "
                    f"but lerobot_frames={episode.lerobot_frames}.",
                    episode_key=key,
                )
            if segment.lerobot_start_frame != previous_lerobot_end + 1:
                report.add(
                    "error",
                    "lerobot_gap_or_overlap",
                    f"Expected start={previous_lerobot_end + 1}, got {segment.lerobot_start_frame}.",
                    episode_key=key,
                )
            previous_lerobot_end = segment.lerobot_end_frame

    if (
        cache_role == TRAIN_CACHE_ROLE
        and episode.lerobot_frames is not None
        and previous_lerobot_end != episode.lerobot_frames - 1
    ):
        report.add(
            "error",
            "lerobot_coverage_incomplete",
            f"Last LeRobot frame must be {episode.lerobot_frames - 1}, got {previous_lerobot_end}.",
            episode_key=key,
        )
    if previous_raw_end != episode.all_frames - 1:
        report.add(
            "error",
            "raw_coverage_incomplete",
            f"Last raw frame must be {episode.all_frames - 1}, got {previous_raw_end}.",
            episode_key=key,
        )


def detect_cache_role(payload: Any) -> CacheRole:
    """Return the explicit cache role, or infer one homogeneous legacy shape."""

    if not isinstance(payload, dict):
        raise PlannerCacheError("Top-level cache payload must be an object.")
    meta = payload.get("meta")
    declared = meta.get("cache_role") if isinstance(meta, dict) else None
    if declared is not None:
        if declared not in VALID_CACHE_ROLES:
            raise PlannerCacheError(
                f"Unsupported cache_role={declared!r}; expected one of {sorted(VALID_CACHE_ROLES)}."
            )
        return declared

    episodes = payload.get("episodes")
    if not isinstance(episodes, dict) or not episodes:
        raise PlannerCacheError("Cannot infer cache role from an empty or invalid episodes object.")
    shapes = {
        TRAIN_CACHE_ROLE
        if isinstance(record, dict) and record.get("output_episode_index") is not None
        else RAW_VALIDATION_CACHE_ROLE
        for record in episodes.values()
    }
    if len(shapes) != 1:
        raise PlannerCacheError("Cache mixes train_lerobot and raw_validation episode shapes.")
    return shapes.pop()


def _iter_episodes(
    payload: dict[str, Any],
    report: PlannerCacheValidationReport,
    *,
    cache_role: CacheRole,
) -> list[PlannerEpisode]:
    episodes_value = payload.get("episodes")
    if not isinstance(episodes_value, dict):
        report.add("error", "invalid_episodes", "Top-level episodes must be an object.")
        return []
    if not episodes_value:
        report.add("error", "empty_cache", "Cache does not contain any episodes.")
        return []

    episodes: list[PlannerEpisode] = []
    seen_episode_indices: set[int] = set()
    for cache_key, record in episodes_value.items():
        key = str(cache_key)
        try:
            episode = _load_episode(key, record)
        except PlannerCacheError as exc:
            report.add("error", "invalid_episode_record", str(exc), episode_key=key)
            continue
        if episode.output_episode_index is not None:
            if episode.output_episode_index in seen_episode_indices:
                report.add(
                    "error",
                    "duplicate_episode_index",
                    f"Duplicate output_episode_index={episode.output_episode_index}.",
                    episode_key=key,
                )
            seen_episode_indices.add(episode.output_episode_index)
        _validate_episode_geometry(episode, report, cache_role=cache_role)
        warning_values = record.get("warnings", [])
        if cache_role == TRAIN_CACHE_ROLE and isinstance(warning_values, list) and any(
            "lerobot_frame_count_mismatch" in str(item) for item in warning_values
        ):
            report.add(
                "error",
                "frame_mapping_warning",
                "Cache record contains lerobot_frame_count_mismatch warning.",
                episode_key=key,
            )
        if bool(record.get("dry_run", False)):
            report.add("error", "dry_run_record", "Dry-run cache records cannot be used for training.", episode_key=key)
        episodes.append(episode)
    report.checked_episodes = len(episodes)
    return episodes


def validate_cache_payload(
    payload: Any,
    *,
    expected_action_space: str = ACTION_SPACE_DELTA,
    expected_cache_role: CacheRole | None = None,
) -> PlannerCacheValidationReport:
    """只校验 cache 自身，不读取任何外部数据集文件。"""

    report = PlannerCacheValidationReport()
    if not isinstance(payload, dict):
        report.add("error", "invalid_payload", "Top-level cache payload must be an object.")
        return report
    try:
        cache_role = detect_cache_role(payload)
    except PlannerCacheError as exc:
        report.add("error", "invalid_cache_role", str(exc))
        return report
    report.cache_role = cache_role
    report.raw_geometry_checked = True
    report.lerobot_geometry_checked = cache_role == TRAIN_CACHE_ROLE
    if expected_cache_role is not None and cache_role != expected_cache_role:
        report.add(
            "error",
            "cache_role_mismatch",
            f"Expected cache_role={expected_cache_role!r}, got {cache_role!r}.",
        )

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        report.add("error", "invalid_meta", "Top-level meta must be an object.")
    else:
        if meta.get("cache_role") is None:
            report.add(
                "error",
                "missing_cache_role",
                "Cache v2 requires an explicit meta.cache_role.",
            )
        if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
            report.add(
                "error",
                "schema_version_mismatch",
                f"Expected schema_version={CACHE_SCHEMA_VERSION!r}, got {meta.get('schema_version')!r}.",
            )
        if meta.get("action_space") != expected_action_space:
            report.add(
                "error",
                "action_space_mismatch",
                f"Expected action_space={expected_action_space!r}, got {meta.get('action_space')!r}.",
            )
        if meta.get("instruction_contract_version") not in (None, INSTRUCTION_CONTRACT_VERSION):
            report.add(
                "error",
                "instruction_contract_mismatch",
                "Cache uses an unsupported instruction contract version.",
            )
        elif meta.get("instruction_contract_version") is None:
            report.add(
                "error",
                "legacy_instruction_contract",
                "Cache predates explicit instruction_contract_version; rebuild before formal M1 training.",
            )
    _iter_episodes(payload, report, cache_role=cache_role)
    return report


def load_conversion_progress(path: str | Path) -> dict[int, dict[str, Any]]:
    """读取 raw episode 到 LeRobot output_episode_index 的转换映射。"""

    progress_path = Path(path).expanduser().resolve()
    if not progress_path.is_file():
        raise FileNotFoundError(f"Conversion progress file does not exist: {progress_path}")
    records: dict[int, dict[str, Any]] = {}
    with progress_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlannerCacheError(
                    f"Invalid JSON at {progress_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise PlannerCacheError(f"Progress record at line {line_number} must be an object.")
            try:
                index = _as_int(record.get("output_episode_index"), "output_episode_index", episode_key="progress")
                frames = _as_int(record.get("frames"), "frames", episode_key="progress")
                raw_key = _as_nonempty_string(record.get("raw_episode_key"), "raw_episode_key", episode_key="progress")
            except PlannerCacheError as exc:
                raise PlannerCacheError(f"Invalid progress record at line {line_number}: {exc}") from exc
            if index in records:
                raise PlannerCacheError(f"Duplicate output_episode_index={index} in {progress_path}.")
            if frames < 1:
                raise PlannerCacheError(f"Progress record {index} has invalid frames={frames}.")
            records[index] = {**record, "output_episode_index": index, "frames": frames, "raw_episode_key": raw_key}
    if not records:
        raise PlannerCacheError(f"Conversion progress file is empty: {progress_path}")
    return records


def validate_against_conversion_progress(
    payload: Any,
    conversion_progress_path: str | Path,
    *,
    require_complete: bool = False,
) -> PlannerCacheValidationReport:
    """校验 cache 的 episode id、raw key 与 LeRobot 帧数是否对应转换产物。"""

    report = validate_cache_payload(payload, expected_cache_role=TRAIN_CACHE_ROLE)
    report.conversion_progress_path = str(Path(conversion_progress_path).expanduser().resolve())
    report.conversion_checked = True
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), dict):
        return report
    if report.cache_role != TRAIN_CACHE_ROLE:
        return report
    try:
        progress_by_index = load_conversion_progress(conversion_progress_path)
    except (FileNotFoundError, PlannerCacheError) as exc:
        report.add("error", "invalid_conversion_progress", str(exc))
        return report

    cache_episode_indices: set[int] = set()
    for cache_key, record in payload["episodes"].items():
        if not isinstance(record, dict):
            continue
        try:
            index = _as_int(record.get("output_episode_index"), "output_episode_index", episode_key=str(cache_key))
        except PlannerCacheError:
            continue
        cache_episode_indices.add(index)
        progress_record = progress_by_index.get(index)
        if progress_record is None:
            report.add(
                "error",
                "episode_missing_from_progress",
                f"output_episode_index={index} is absent from conversion progress.",
                episode_key=str(cache_key),
            )
            continue
        if record.get("raw_episode_key") != progress_record["raw_episode_key"]:
            report.add(
                "error",
                "raw_episode_key_mismatch",
                f"Cache raw_episode_key={record.get('raw_episode_key')!r} does not match progress "
                f"{progress_record['raw_episode_key']!r}.",
                episode_key=str(cache_key),
            )
        if record.get("lerobot_frames") != progress_record["frames"]:
            report.add(
                "error",
                "lerobot_frames_mismatch",
                f"Cache lerobot_frames={record.get('lerobot_frames')!r} does not match progress "
                f"{progress_record['frames']!r}.",
                episode_key=str(cache_key),
            )

    if require_complete:
        missing_indices = sorted(set(progress_by_index) - cache_episode_indices)
        for index in missing_indices:
            report.add(
                "error",
                "episode_missing_from_cache",
                f"Conversion episode {index} ({progress_by_index[index]['raw_episode_key']}) is missing from cache.",
                episode_key=str(index),
            )
    # A structurally valid partial cache is useful for smoke tests, but it must
    # not be advertised as training-ready without an explicit completeness gate.
    report.training_eligible = report.is_valid and require_complete
    return report


class PlannerCacheIndex:
    """训练/推理侧的只读 planner-cache 查询器。"""

    def __init__(self, payload: dict[str, Any], *, source_path: str | Path | None = None) -> None:
        report = validate_cache_payload(payload, expected_cache_role=TRAIN_CACHE_ROLE)
        if not report.is_valid:
            raise PlannerCacheValidationError(report)
        episodes = _iter_episodes(
            payload,
            PlannerCacheValidationReport(),
            cache_role=TRAIN_CACHE_ROLE,
        )
        self._episodes_by_index = {
            episode.output_episode_index: episode
            for episode in episodes
            if episode.output_episode_index is not None
        }
        self.source_path = str(Path(source_path).expanduser().resolve()) if source_path is not None else None
        self.metadata = dict(payload["meta"])
        self.fingerprint = cache_fingerprint(payload)

    @classmethod
    def from_path(cls, path: str | Path) -> "PlannerCacheIndex":
        cache_path = Path(path).expanduser().resolve()
        if not cache_path.is_file():
            raise FileNotFoundError(f"Planner cache file does not exist: {cache_path}")
        with cache_path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise PlannerCacheError(f"Invalid planner-cache JSON: {cache_path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise PlannerCacheError(f"Planner cache top-level value must be an object: {cache_path}")
        return cls(payload, source_path=cache_path)

    @property
    def episode_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._episodes_by_index))

    def resolve(self, episode_index: int, frame_index: int) -> SubtaskContext:
        """严格查询当前 LeRobot 样本所属的唯一子任务。"""

        if isinstance(episode_index, bool) or isinstance(frame_index, bool):
            raise TypeError("episode_index and frame_index must be integers, not bool.")
        episode_index = int(episode_index)
        frame_index = int(frame_index)
        episode = self._episodes_by_index.get(episode_index)
        if episode is None:
            raise KeyError(
                f"Planner cache has no record for output_episode_index={episode_index}; "
                "training must not fall back to the full-task prompt."
            )
        if not 0 <= frame_index < episode.lerobot_frames:
            raise IndexError(
                f"frame_index={frame_index} is outside episode {episode_index} range "
                f"[0, {episode.lerobot_frames - 1}]."
            )
        matches = [segment for segment in episode.segments if segment.contains_lerobot_frame(frame_index)]
        if len(matches) != 1:
            raise PlannerCacheError(
                f"Expected exactly one segment for episode={episode_index}, frame={frame_index}; got {len(matches)}."
            )
        segment = matches[0]
        if segment.lerobot_start_frame is None or segment.lerobot_end_frame is None:
            raise PlannerCacheError(
                f"Episode={episode_index}, segment={segment.segment_index} lacks LeRobot frame geometry."
            )
        return SubtaskContext(
            episode_index=episode_index,
            frame_index=frame_index,
            raw_episode_key=episode.raw_episode_key,
            segment_index=segment.segment_index,
            action=segment.action,
            instruction=segment.instruction,
            code_policy=segment.code_policy,
            lerobot_start_frame=segment.lerobot_start_frame,
            lerobot_end_frame=segment.lerobot_end_frame,
            cache_fingerprint=self.fingerprint,
        )

    def resolve_from_sample_metadata(
        self,
        sample_metadata: Mapping[str, Any],
        *,
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
    ) -> SubtaskContext:
        """从 DataLoader 暴露的元数据严格解析子任务上下文。

        Phase 3 的 LeRobot 包装层会把底层样本的 episode/frame 字段传入这里。
        不提供默认值，避免元数据缺失时悄悄使用错误 instruction。
        """

        if episode_key not in sample_metadata:
            raise KeyError(f"Sample metadata is missing {episode_key!r}.")
        if frame_key not in sample_metadata:
            raise KeyError(f"Sample metadata is missing {frame_key!r}.")
        return self.resolve(sample_metadata[episode_key], sample_metadata[frame_key])


def cache_fingerprint(payload: Any) -> str:
    """对 JSON 内容做稳定 SHA256，供 checkpoint 与评测结果绑定输入 cache。"""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_issues(issues: Iterable[ValidationIssue]) -> str:
    """生成适合 CLI 的简短问题摘要。"""

    return "; ".join(
        f"{issue.level}:{issue.code}" + (f"[{issue.episode_key}]" if issue.episode_key else "")
        for issue in issues
    )
