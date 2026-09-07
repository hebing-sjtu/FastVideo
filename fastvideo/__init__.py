import os

# ``python -m fastvideo...`` imports this file before the training entrypoint.
# ``checkpoint_wrapper`` (NO_REENTRANT) compiles through inductor; on A3-Ultra
# H200 that kernel is ``CUDA driver error: invalid argument``. Set this before
# any ``import torch``.
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.version import __version__

__all__ = ["VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"]
