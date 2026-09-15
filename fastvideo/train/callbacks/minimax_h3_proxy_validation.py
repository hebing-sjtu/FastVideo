# SPDX-License-Identifier: Apache-2.0
"""Validation for MiniMax-H3 proxy-to-video, logging a side-by-side comparison.

The generic :class:`~fastvideo.train.callbacks.validation.ValidationCallback` conditions a pipeline
through ``sampling_param.image_path``, which H3 Ref2VA does not read: it takes an *ordered* list of
references, and their order is load-bearing because the text was tokenized around ``<Picture 1>``
then ``<Video 1>`` labels. So this subclass builds ``batch.references`` itself.

It also fills the base's second video stream with a proxy / prediction / target panel instead of an
action overlay. A prediction on its own cannot answer the question this stage is being trained on --
whether the output follows the proxy -- and asking a viewer to scrub three players in sync makes
that comparison something they do rather than something the tracker records.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.train.callbacks.validation import ValidationCallback

logger = init_logger(__name__)

# Read from each validation record. `proxy_path` is required; the rest are optional, and each one
# missing removes a panel or a conditioning slot rather than failing the run.
PROXY_PATH_KEY = "proxy_path"
TARGET_PATH_KEY = "target_path"
ANCHOR_PATH_KEY = "anchor_path"
CAMERA_PATH_KEY = "camera_path"


class MiniMaxH3ProxyValidationCallback(ValidationCallback):
    """Sample H3 Ref2VA from an anchor plus a proxy render, logging a comparison panel."""

    _secondary_video_suffix = "_compare"
    _secondary_video_key_suffix = "_compare"

    def __init__(
        self,
        *,
        panel_labels: bool = True,
        panel_separator_px: int = 4,
        panel_height: int | None = None,
        include_target_panel: bool = True,
        anchor_short_edge: int = 2048,
        proxy_height: int = 192,
        proxy_width: int = 336,
        cwm_system_prompt: str = "w0",
        num_given_latent_frames: int = 1,
        lock_first_frame: bool | None = None,
        **kwargs: Any,
    ) -> None:
        # Before the base, which loads a pipeline: a contradiction between the given-frame count and
        # the system prompt is decidable from the arguments alone, and is cheaper to report than to
        # discover after a model is resident.
        #
        # Must match the model plugin's `num_given_latent_frames` *and* the CWM system prompt the
        # cache's text was wrapped in: 1 is w0's "the first frame is locked to this image", 10 is
        # wn's "the first 34 frames are ALREADY GIVEN". Sampling with fewer given frames than
        # training supervised under leaves the model inventing rows it was never asked to predict.
        if lock_first_frame is not None:
            raise ValueError("lock_first_frame was replaced by num_given_latent_frames, which has to say how many "
                             f"frames are given rather than whether any are: pass num_given_latent_frames="
                             f"{1 if lock_first_frame else 0} for the old lock_first_frame={lock_first_frame} "
                             "behaviour, or 10 to sample CWM's wn regime.")
        self.num_given_latent_frames = int(num_given_latent_frames)
        if self.num_given_latent_frames < 0:
            raise ValueError(f"num_given_latent_frames must be non-negative, got {num_given_latent_frames!r}")
        role = str(cwm_system_prompt or "none")
        if self.num_given_latent_frames > 1 and role != "wn":
            raise ValueError(f"num_given_latent_frames={self.num_given_latent_frames} is CWM's wn regime, but "
                             f"cwm_system_prompt={role!r}. The w0 prompt tells the model this clip is the very "
                             "beginning of the take with nothing preceding it, which contradicts handing it footage "
                             "to continue; set cwm_system_prompt: wn and encode the cache to match.")
        # And the other direction, which is the one a step-0 baseline falls into: the eval wrapper
        # defaults to a w0 config, so pointing it at a wn cache samples one given frame against a
        # prompt that says thirty-four are already given. The prompt wins, the rows contradict it,
        # and the panel reads as a bad checkpoint rather than as a misconfigured run.
        if role == "wn" and self.num_given_latent_frames <= 1:
            raise ValueError(f"cwm_system_prompt='wn' promises the model that 'The first 34 frames (1.4167 seconds) "
                             f"of this clip are ALREADY GIVEN', but num_given_latent_frames="
                             f"{self.num_given_latent_frames} hands it none of them. Set "
                             "num_given_latent_frames: 10 (CWM's VIDEO_PREFIX_LATENTS), or sample a w0 cache with "
                             "cwm_system_prompt: w0.")

        super().__init__(**kwargs)
        self.panel_labels = self._coerce_bool(panel_labels)
        self.panel_separator_px = int(panel_separator_px)
        # Three 768-tall panels side by side is a ~4000px-wide artifact per validation event. The
        # panel is for eyeballing whether the output tracks the proxy, which survives downscaling;
        # the prediction is also logged at full size on its own.
        self.panel_height = int(panel_height) if panel_height else None
        if self.panel_height is not None and self.panel_height <= 0:
            raise ValueError(f"panel_height must be positive, got {panel_height}.")
        self.include_target_panel = self._coerce_bool(include_target_panel)
        # Must equal the encoder's --anchor-short-edge, which defaults to the same released 2048.
        # The anchor's canvas decides how many vision tokens it occupies, so a mismatch presents the
        # model a token grid it never trained on and validation looks worse than the checkpoint is,
        # for a reason that has nothing to do with the checkpoint.
        self.anchor_short_edge = int(anchor_short_edge)
        if self.anchor_short_edge <= 0:
            raise ValueError(f"anchor_short_edge must be positive, got {anchor_short_edge}.")
        # Must equal the encoder's --proxy-height/--proxy-width, for the same reason as the anchor
        # and then some. Left unpinned, a video reference resolves its canvas from its aspect ratio,
        # which puts the 336x192 proxy this stage trains on onto the full 1344x768 canvas: 37296
        # reference rows instead of 2442, from LANCZOS-upsampled frames. For a DUV proxy that
        # resampling is not merely off-distribution, it averages unrelated depth codes.
        self.proxy_size = (int(proxy_height), int(proxy_width))
        if min(self.proxy_size) <= 0:
            raise ValueError(f"proxy_height and proxy_width must be positive, got {self.proxy_size}.")
        # Must match how the training cache's text embedding was wrapped. ABot clips are window 0.
        self.cwm_system_prompt = role
        # The base gates its second video stream on `overlay_actions`. Nothing about that plumbing
        # is overlay-specific -- it saves, gathers across sequence-parallel groups, and logs under
        # its own key -- so the comparison panel rides it rather than duplicating the 170-line
        # method that owns it.
        self.overlay_actions = True
        if self.use_validation_media_conditioning:
            raise ValueError("MiniMaxH3ProxyValidationCallback requires use_validation_media_conditioning=false. H3 "
                             "Ref2VA conditions on an ordered reference list, not on "
                             "sampling_param.image_path, and setting both would present the proxy twice.")
        if self.num_videos_per_prompt != 1:
            raise ValueError("H3 packs one document per request, so num_videos_per_prompt must be 1; got "
                             f"{self.num_videos_per_prompt}.")
        self._pending_record: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def _prepare_validation_batch(
        self,
        sampling_param: Any,
        validation_batch: dict[str, Any],
        num_inference_steps: int,
    ) -> ForwardBatch:
        """Attach the anchor and proxy as ordered Ref2VA references."""
        from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference

        batch = super()._prepare_validation_batch(
            sampling_param,
            validation_batch,
            num_inference_steps,
        )
        proxy_path = self._record_path(validation_batch, PROXY_PATH_KEY)
        if proxy_path is None:
            raise ValueError(f"Every H3 proxy validation record needs a readable {PROXY_PATH_KEY!r}; got "
                             f"{validation_batch.get(PROXY_PATH_KEY)!r}. Paths must be absolute: the loader only "
                             "resolves the media keys it knows about against the dataset directory.")

        # Anchor first, proxy second, matching how the training cache was encoded. The anchor is
        # what fixes appearance, which a proxy render cannot supply.
        batch.references = [
            MiniMaxH3Reference(
                source=self._resolve_anchor(validation_batch),
                media_type="image",
                short_edge=self.anchor_short_edge,
            ),
            MiniMaxH3Reference(source=proxy_path, media_type="video", size=self.proxy_size),
        ]

        camera = self._record_path(validation_batch, CAMERA_PATH_KEY)
        if camera is not None:
            from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_camera_conditioning import (
                MINIMAX_H3_CAMERA_TRAJECTORY_KEY, )
            batch.extra[MINIMAX_H3_CAMERA_TRAJECTORY_KEY] = camera

        from fastvideo.pipelines.basic.minimax_h3.cwm_presentation import CWM_SYSTEM_PROMPT_KEY

        if self.cwm_system_prompt and self.cwm_system_prompt != "none":
            batch.extra[CWM_SYSTEM_PROMPT_KEY] = self.cwm_system_prompt

        if self.num_given_latent_frames:
            from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import (
                MINIMAX_H3_GIVEN_FRAMES_KEY,
                MINIMAX_H3_NUM_GIVEN_LATENT_FRAMES_KEY,
            )
            batch.extra[MINIMAX_H3_NUM_GIVEN_LATENT_FRAMES_KEY] = self.num_given_latent_frames
            if self.num_given_latent_frames > 1:
                batch.extra[MINIMAX_H3_GIVEN_FRAMES_KEY] = self._given_frames(validation_batch)

        # Held for `_post_process_validation_frames`, which the base calls later in the same loop
        # iteration and does not pass the record to.
        self._pending_record = dict(validation_batch)
        return batch

    def _resolve_anchor(self, record: dict[str, Any]) -> Any:
        """The anchor image, defaulting to the target's first frame as the encoder does.

        A video path cannot be handed over as an image reference, because an image source that is a
        path goes through ``load_image``. The decoded frame is passed as a PIL image instead.
        """
        anchor = self._record_path(record, ANCHOR_PATH_KEY)
        if anchor is not None:
            return anchor
        target = self._record_path(record, TARGET_PATH_KEY)
        if target is None:
            raise ValueError(f"Record needs {ANCHOR_PATH_KEY!r} or {TARGET_PATH_KEY!r} to supply the appearance "
                             "anchor; a proxy alone leaves the model nothing to take appearance from.")
        from PIL import Image

        from fastvideo.pipelines.basic.minimax_h3.reference import (
            decode_reference_video, )
        frames, _, _ = decode_reference_video(target)
        return Image.fromarray(frames[0])

    def _given_frames(self, record: dict[str, Any]) -> np.ndarray:
        """The target clip on H3's 24-fps timeline, to be encoded and sliced for the given prefix.

        The whole clip, not its first 34 frames: the latent stage encodes it and keeps the leading
        latent frames, because the VAE's group structure cannot produce 10 latent frames from a
        standalone encode and slicing a full encode is what training does to the cached target.

        Using the target's own opening is the validation analogue of CWM's windowed inference, where
        every window past the first continues the previous window's decoded output. It does mean
        this regime cannot be evaluated without a target, which is the point: the prefix is real
        footage by definition.
        """
        target = self._record_path(record, TARGET_PATH_KEY)
        if target is None:
            raise ValueError(f"num_given_latent_frames={self.num_given_latent_frames} needs {TARGET_PATH_KEY!r} to "
                             "supply the given footage; the record has none. Only a one-frame prefix can be taken "
                             "from the anchor.")
        from fastvideo.pipelines.basic.minimax_h3.reference import (
            decode_reference_video,
            resample_reference_frames,
        )

        frames, source_fps, _ = decode_reference_video(target)
        return resample_reference_frames(frames, source_fps)

    @staticmethod
    def _record_path(record: dict[str, Any], key: str) -> str | None:
        """A readable path for ``key``, or None.

        Absent, null, and unreadable are treated alike. ``datasets`` gives every record the union
        of all keys, so a field only some rows carry arrives as None on the others, and a column
        that is null everywhere loses its type entirely.
        """
        value = record.get(key)
        if not isinstance(value, str) or not value:
            return None
        return value if os.path.isfile(value) else None

    # ------------------------------------------------------------------
    # Comparison panel
    # ------------------------------------------------------------------

    def _post_process_validation_frames(
        self,
        frames: list[np.ndarray],
        *,
        action: dict[str, Any] | None,
    ) -> list[np.ndarray] | None:
        """Compose proxy | prediction | target at the prediction's resolution."""
        del action
        record, self._pending_record = self._pending_record, None
        if record is None or not frames:
            return None

        from fastvideo.train.utils.video_panels import compose_comparison_video

        height = self.panel_height or int(frames[0].shape[0])
        num_frames = len(frames)
        panels: list[list[np.ndarray]] = []
        labels: list[str] = []

        proxy = self._read_panel(record, PROXY_PATH_KEY, num_frames)
        if proxy is not None:
            panels.append(proxy)
            labels.append("proxy (src)")
        panels.append(frames)
        labels.append("prediction")
        if self.include_target_panel:
            target = self._read_panel(record, TARGET_PATH_KEY, num_frames)
            if target is not None:
                panels.append(target)
                labels.append("target (tgt)")

        if len(panels) < 2:
            # Only the prediction resolved, which the base already logs on its own.
            return None
        try:
            return compose_comparison_video(
                panels,
                labels=labels if self.panel_labels else None,
                separator_px=self.panel_separator_px,
                height=height,
            )
        except (ValueError, OSError) as error:
            # Comparison media is diagnostic, so a bad panel costs the panel, not the run.
            logger.warning("Skipping H3 validation comparison panel: %s", error)
            return None

    def _read_panel(self, record: dict[str, Any], key: str, num_frames: int) -> list[np.ndarray] | None:
        """Decode a comparison clip onto H3's 24-fps timeline."""
        path = self._record_path(record, key)
        if path is None:
            return None
        from fastvideo.pipelines.basic.minimax_h3.reference import (
            decode_reference_video,
            resample_reference_frames,
        )

        try:
            decoded, source_fps, _ = decode_reference_video(path)
            # Resample before trimming, the same order the encoder used. Trimming first would show
            # a different span of the clip than the model was conditioned on whenever the source is
            # not already at 24 fps.
            decoded = resample_reference_frames(decoded, source_fps)
        except Exception as error:  # noqa: BLE001 - a container can fail in codec-specific ways
            logger.warning("Could not decode %s for the validation panel: %s", path, error)
            return None
        if decoded.shape[0] == 0:
            return None
        return list(decoded[:num_frames])


__all__ = ["MiniMaxH3ProxyValidationCallback"]
