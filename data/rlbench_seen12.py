"""Seen12 训练数据契约。

这里不复制 episode 划分或 norm-stats 路径。运行时只从
``openpi.training.config`` 读取两者，并生成可写入 planner-cache 的稳定指纹。
模块本身不在 import 时加载 OpenPI，便于 cache 工具和标准库单元测试使用。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENPI_ROOT = REPO_ROOT / "openpi-vqap"
DEFAULT_TRAIN_CONFIG_NAME = "pi05_rlbench_delta_seen12_pt"
EXPECTED_REPO_ID = "train_delta"
EXPECTED_ASSET_ID = "train_delta_seen12"
EXPECTED_EPISODE_COUNT = 1200
NORM_STAT_FIELDS = ("mean", "std", "q01", "q99")


@dataclass(frozen=True)
class Seen12DataContract:
    """训练 cache 必须绑定的 Seen12 配置快照。"""

    config_name: str
    repo_id: str
    asset_id: str
    episode_indices: tuple[int, ...]
    norm_stats_path: Path
    episode_indices_sha256: str
    norm_stats_sha256: str

    def cache_metadata(self) -> dict[str, Any]:
        return {
            "train_config_name": self.config_name,
            "lerobot_repo_id": self.repo_id,
            "norm_stats_asset_id": self.asset_id,
            "expected_episode_count": len(self.episode_indices),
            "episode_indices_sha256": self.episode_indices_sha256,
            "norm_stats_sha256": self.norm_stats_sha256,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_indices_fingerprint(indices: Iterable[int]) -> str:
    """对有序 episode 列表生成跨进程稳定的指纹。"""

    encoded = json.dumps(
        [int(index) for index in indices],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_config_module(openpi_root: Path) -> ModuleType:
    source_root = openpi_root / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"OpenPI source root not found: {source_root}")
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return importlib.import_module("openpi.training.config")


def _resolve_norm_stats_path(*, openpi_root: Path, assets: Any, asset_id: str) -> Path:
    assets_dir_value = getattr(assets, "assets_dir", None)
    if not assets_dir_value:
        raise ValueError("Seen12 config must define data.assets.assets_dir explicitly.")
    assets_dir = Path(str(assets_dir_value)).expanduser()
    if not assets_dir.is_absolute():
        assets_dir = openpi_root / assets_dir
    return (assets_dir / asset_id / "norm_stats.json").resolve()


def _validate_norm_stats(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Seen12 norm stats not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Seen12 norm stats JSON: {path}: {exc.msg}") from exc
    stats = payload.get("norm_stats") if isinstance(payload, dict) else None
    if not isinstance(stats, dict):
        raise ValueError(f"{path} must contain a norm_stats object.")
    for key in ("state", "actions"):
        group = stats.get(key)
        if not isinstance(group, dict):
            raise ValueError(f"{path}: norm_stats.{key} must be an object.")
        lengths: set[int] = set()
        for field in NORM_STAT_FIELDS:
            values = group.get(field)
            if not isinstance(values, list) or not values:
                raise ValueError(f"{path}: norm_stats.{key}.{field} must be a non-empty list.")
            lengths.add(len(values))
            for position, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"{path}: norm_stats.{key}.{field}[{position}] must be numeric."
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"{path}: norm_stats.{key}.{field}[{position}] must be finite."
                    )
        if len(lengths) != 1:
            raise ValueError(f"{path}: norm_stats.{key} field dimensions disagree: {lengths}.")


def load_seen12_data_contract(
    config_name: str = DEFAULT_TRAIN_CONFIG_NAME,
    *,
    openpi_root: str | Path = DEFAULT_OPENPI_ROOT,
    config_module: ModuleType | Any | None = None,
) -> Seen12DataContract:
    """从 OpenPI 唯一事实源读取并严格校验 Seen12 训练契约。"""

    root = Path(openpi_root).expanduser().resolve()
    module = config_module if config_module is not None else _load_config_module(root)
    config = module.get_config(config_name)
    source_indices = tuple(int(index) for index in module.RLBENCH_SEEN12_EPISODE_INDICES)
    configured_indices = tuple(int(index) for index in getattr(config.data, "episode_indices", ()))

    if getattr(config, "name", None) != config_name:
        raise ValueError(
            f"Requested config={config_name!r}, but source returned {getattr(config, 'name', None)!r}."
        )
    if configured_indices != source_indices:
        raise ValueError(
            "Seen12 TrainConfig episode_indices differs from "
            "RLBENCH_SEEN12_EPISODE_INDICES."
        )
    if len(source_indices) != EXPECTED_EPISODE_COUNT:
        raise ValueError(
            f"Seen12 must contain {EXPECTED_EPISODE_COUNT} episodes, got {len(source_indices)}."
        )
    if len(set(source_indices)) != len(source_indices):
        raise ValueError("Seen12 episode indices contain duplicates.")
    if any(index < 0 for index in source_indices):
        raise ValueError("Seen12 episode indices must be non-negative.")

    repo_id = str(getattr(config.data, "repo_id", "") or "")
    if repo_id != EXPECTED_REPO_ID:
        raise ValueError(f"Seen12 repo_id must be {EXPECTED_REPO_ID!r}, got {repo_id!r}.")
    assets = getattr(config.data, "assets", None)
    if assets is None:
        raise ValueError("Seen12 config must define data.assets.")
    asset_id = str(getattr(assets, "asset_id", "") or "")
    if asset_id != EXPECTED_ASSET_ID:
        raise ValueError(f"Seen12 asset_id must be {EXPECTED_ASSET_ID!r}, got {asset_id!r}.")

    norm_stats_path = _resolve_norm_stats_path(
        openpi_root=root,
        assets=assets,
        asset_id=asset_id,
    )
    _validate_norm_stats(norm_stats_path)
    return Seen12DataContract(
        config_name=config_name,
        repo_id=repo_id,
        asset_id=asset_id,
        episode_indices=source_indices,
        norm_stats_path=norm_stats_path,
        episode_indices_sha256=episode_indices_fingerprint(source_indices),
        norm_stats_sha256=_sha256_file(norm_stats_path),
    )
