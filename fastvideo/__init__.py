import os

# ``python -m fastvideo...`` imports this file before the training entrypoint.
# AOT/inductor is not torch.compile: TORCHINDUCTOR_DISABLE is not a PyTorch
# switch. These two are. Set them before any ``import torch``.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator
from fastvideo.version import __version__

__all__ = ["VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"]
