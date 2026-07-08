"""RLBench 评测录像工具：环绕相机逐仿真步抓帧，按成功/失败配额流式写 mp4。

设计要点：
- 相机挂在场景内置的 cam_cinematic_placeholder 上（RLBench 官方 cinematic_recorder 同款），
  开启 explicit handling，只在真正抓帧时渲染，不录制时零开销。
- 通过 scene.register_step_callback 逐仿真步抓帧，保证 planning 类 action 的整条路径流畅。
- 帧直接流式写入临时 mp4（不在内存攒帧），episode 结束后按成功/失败与配额决定改名保留或删除。
- pyrep / cv2 均延迟导入：pyrep 依赖仿真进程，cv2 的 Qt 与 PyRep 存在插件冲突。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


class CircleCameraMotion:
    """让挂在 cam_cinematic_placeholder 上的相机绕场景基点缓慢环绕；speed=0 时为固定机位。"""

    def __init__(self, cam, origin, speed: float):
        self.cam = cam
        self.origin = origin
        self.speed = float(speed)
        self._initial_pose = None

    def save_pose(self) -> None:
        self._initial_pose = self.cam.get_pose()

    def restore_pose(self) -> None:
        if self._initial_pose is not None:
            self.cam.set_pose(self._initial_pose)

    def step(self) -> None:
        if self.speed != 0.0:
            self.origin.rotate([0.0, 0.0, self.speed])


class EpisodeVideoRecorder:
    """按任务分别录制前 N 条成功与前 N 条失败 episode 的视频。

    调用顺序：
        recorder.attach(scene)                    # env.launch() 之后调用一次
        recorder.start_episode(...)               # 每条 episode reset 之后
        path = recorder.end_episode(success=...)  # episode 结束；返回保留的视频路径或 None
        recorder.finalize()                       # 评测收尾/异常中断时释放资源、清理临时文件
        recorder.write_summary(...)               # 全部结束后生成 recording_summary.log
    """

    def __init__(
        self,
        save_root: Path,
        *,
        videos_per_task: int = 1,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        rotate_speed: float = 0.005,
        logger: logging.Logger | None = None,
    ):
        if videos_per_task < 1:
            raise ValueError(f"videos_per_task must be >= 1, got {videos_per_task}")
        self._save_root = Path(save_root)
        self._quota = int(videos_per_task)
        self._resolution = (int(resolution[0]), int(resolution[1]))
        self._fps = int(fps)
        self._rotate_speed = float(rotate_speed)
        self._logger = logger or logging.getLogger(__name__)

        self._cam_motion: CircleCameraMotion | None = None
        self._writer = None
        self._tmp_path: Path | None = None
        self._frames_written = 0
        self._capturing = False
        self._meta: dict[str, Any] | None = None
        # task -> {episodes, successes, success_videos, fail_videos}，按评测顺序插入。
        self._task_stats: dict[str, dict[str, Any]] = {}

    """创建录像相机并注册逐仿真步回调；必须在 env.launch() 之后调用一次。"""
    def attach(self, scene) -> None:
        from pyrep.const import RenderMode
        from pyrep.objects.dummy import Dummy
        from pyrep.objects.vision_sensor import VisionSensor

        cam_placeholder = Dummy("cam_cinematic_placeholder")
        cam = VisionSensor.create(list(self._resolution))
        cam.set_explicit_handling(True)
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)
        cam.set_render_mode(RenderMode.OPENGL)
        self._cam_motion = CircleCameraMotion(cam, Dummy("cam_cinematic_base"), self._rotate_speed)
        self._cam_motion.save_pose()
        scene.register_step_callback(self._on_sim_step)
        self._save_root.mkdir(parents=True, exist_ok=True)
        self._logger.info(
            "Video recorder attached | save_root=%s videos_per_task=%d(success/fail each) "
            "resolution=%dx%d fps=%d rotate_speed=%.4f",
            self._save_root,
            self._quota,
            self._resolution[0],
            self._resolution[1],
            self._fps,
            self._rotate_speed,
        )

    def _stats(self, task_name: str) -> dict[str, Any]:
        return self._task_stats.setdefault(
            task_name,
            {"episodes": 0, "successes": 0, "success_videos": [], "fail_videos": []},
        )

    """episode reset 之后调用：登记 episode 并在配额未满时开始抓帧（写入临时文件）。"""
    def start_episode(self, *, task_name: str, prompt: str, variation: int, local_episode: int) -> None:
        if self._cam_motion is None:
            raise RuntimeError("attach() must be called before start_episode().")
        stats = self._stats(task_name)
        stats["episodes"] += 1
        self._meta = {
            "task": task_name,
            "prompt": " ".join(str(prompt).split()),
            "variation": int(variation),
            "local_episode": int(local_episode),
        }

        need_more = (
            len(stats["success_videos"]) < self._quota
            or len(stats["fail_videos"]) < self._quota
        )
        if not need_more:
            self._capturing = False
            return

        import cv2

        task_dir = self._save_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_path = task_dir / f"var{variation}_ep{local_episode}.tmp.mp4"
        self._writer = cv2.VideoWriter(
            str(self._tmp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            self._resolution,
        )
        self._frames_written = 0
        self._cam_motion.restore_pose()
        self._capturing = True

    """scene 每个仿真步的回调：旋转相机、渲染一帧、叠印 prompt 后写入视频。"""
    def _on_sim_step(self) -> None:
        if not self._capturing or self._writer is None:
            return
        import cv2
        import numpy as np

        self._cam_motion.step()
        self._cam_motion.cam.handle_explicitly()
        frame = (self._cam_motion.cam.capture_rgb() * 255.0).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = self._overlay_prompt(cv2, frame)
        self._writer.write(frame)
        self._frames_written += 1

    """把语言指令叠印在画面底部（黑描边+白字，参考 TAVP video_utils）。"""
    def _overlay_prompt(self, cv2_mod, frame):
        prompt = self._meta.get("prompt", "") if self._meta else ""
        if not prompt:
            return frame
        width, height = self._resolution
        text = prompt if len(prompt) <= 80 else prompt[:77] + "..."
        font = cv2_mod.FONT_HERSHEY_DUPLEX
        font_scale = 0.45 * width / 640
        thickness = max(1, round(width / 640))
        text_size = cv2_mod.getTextSize(text, font, font_scale, thickness)[0]
        org = (max(0, (width - text_size[0]) // 2), height - 15)
        frame = cv2_mod.putText(
            frame, text, org, font, font_scale, (0, 0, 0), thickness + 2, cv2_mod.LINE_AA
        )
        return cv2_mod.putText(
            frame, text, org, font, font_scale, (255, 255, 255), thickness, cv2_mod.LINE_AA
        )

    """episode 结束后调用：按成功/失败与配额决定保留（返回最终路径）或丢弃（返回 None）。"""
    def end_episode(self, *, success: bool, error_type: str | None = None) -> Path | None:
        if self._meta is None:
            return None
        meta = self._meta
        self._meta = None
        stats = self._stats(meta["task"])
        stats["successes"] += int(success)

        capturing = self._capturing
        self._capturing = False
        writer = self._writer
        self._writer = None
        tmp_path = self._tmp_path
        self._tmp_path = None
        if writer is not None:
            writer.release()
        if not capturing or tmp_path is None:
            return None

        bucket = stats["success_videos"] if success else stats["fail_videos"]
        if self._frames_written <= 0 or len(bucket) >= self._quota:
            tmp_path.unlink(missing_ok=True)
            return None

        error_suffix = f"_{error_type}" if (not success and error_type) else ""
        final_name = (
            f"{'success' if success else 'fail'}"
            f"_var{meta['variation']}_ep{meta['local_episode']}{error_suffix}.mp4"
        )
        final_path = tmp_path.with_name(final_name)
        tmp_path.rename(final_path)
        bucket.append(final_name)
        return final_path

    """释放 writer 并删除残留临时文件；评测收尾或异常中断时调用，可重复调用。"""
    def finalize(self) -> None:
        self._capturing = False
        self._meta = None
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._tmp_path is not None:
            self._tmp_path.unlink(missing_ok=True)
            self._tmp_path = None

    """汇总录制情况并写入 save_root/recording_summary.log，返回其路径。"""
    def write_summary(self, *, header: str = "") -> Path:
        items = list(self._task_stats.items())
        total_success = sum(len(stats["success_videos"]) for _, stats in items)
        total_fail = sum(len(stats["fail_videos"]) for _, stats in items)

        lines: list[str] = []
        if header:
            lines.append(header)
        lines.append(
            f"Recording summary | videos_per_task={self._quota} (success/fail each) "
            f"resolution={self._resolution[0]}x{self._resolution[1]} fps={self._fps}"
        )
        lines.append(
            f"Total videos recorded: {total_success + total_fail} "
            f"(success={total_success}, fail={total_fail})"
        )
        lines.append("")
        lines.append("Per-task detail:")
        for task_name, stats in items:
            lines.append(
                f"  {task_name:<32} | episodes={stats['episodes']:<3d} "
                f"successes={stats['successes']:<3d} | "
                f"success_videos={len(stats['success_videos'])} "
                f"fail_videos={len(stats['fail_videos'])}"
            )
            for name in stats["success_videos"] + stats["fail_videos"]:
                lines.append(f"    - {task_name}/{name}")

        with_success = [task for task, stats in items if stats["success_videos"]]
        without_success = [task for task, stats in items if not stats["success_videos"]]
        without_fail = [task for task, stats in items if not stats["fail_videos"]]
        lines.append("")
        lines.append(
            f"Tasks WITH success video ({len(with_success)}): {', '.join(with_success) or '-'}"
        )
        lines.append(
            f"Tasks WITHOUT success video (no successful episode) ({len(without_success)}): "
            f"{', '.join(without_success) or '-'}"
        )
        lines.append(
            f"Tasks without fail video (all episodes succeeded) ({len(without_fail)}): "
            f"{', '.join(without_fail) or '-'}"
        )

        self._save_root.mkdir(parents=True, exist_ok=True)
        summary_path = self._save_root / "recording_summary.log"
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary_path
