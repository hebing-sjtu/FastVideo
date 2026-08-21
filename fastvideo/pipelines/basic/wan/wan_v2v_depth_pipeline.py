# SPDX-License-Identifier: Apache-2.0
"""Inference pipeline for Wan video-to-video with a depth ControlNet.

This is the evaluation counterpart to the ``WanV2VDepthModel`` training plugin.
It differs from ``WanVideoToVideoPipeline`` in two ways that matter:

* Conditioning goes through ``WanV2VDepthConditioningStage``, which reproduces
  the training plugin's ``[noise | mask | source]`` channel layout instead of
  the Wan-Fun-Control ``[noise | source | zeros]`` one.
* No CLIP image encoder is loaded. The Fun-InP and A14B snapshots do ship one,
  and their transformers keep an image cross-attention branch, but the training
  plugin never populates ``encoder_hidden_states_image`` -- the cached samples
  carry no CLIP feature. Leaving the encoder out of the required modules is what
  guarantees validation runs that branch in the same inactive state training
  did, rather than depending on a stage happening not to fill it in.
"""

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (FlowUniPCMultistepScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (ConditioningStage, DecodingStage, DenoisingStage, InputValidationStage,
                                        LatentPreparationStage, TextEncodingStage, TimestepPreparationStage,
                                        WanV2VDepthConditioningStage)
# isort: on

logger = init_logger(__name__)


class WanV2VDepthPipeline(LoRAPipeline, ComposedPipelineBase):
    """Wan video-to-video + depth ControlNet sampling pipeline."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with proper dependency injection."""

        self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(scheduler=self.get_module("scheduler"),
                                                    transformer=self.get_module("transformer")))

        # Runs after latent preparation so the conditioning latent is built
        # against the same frame count the noise tensor was allocated for.
        self.add_stage(stage_name="v2v_depth_conditioning_stage",
                       stage=WanV2VDepthConditioningStage(
                           vae=self.get_module("vae"),
                           transformer=self.get_module("transformer"),
                       ))

        self.add_stage(stage_name="denoising_stage",
                       stage=DenoisingStage(transformer=self.get_module("transformer"),
                                            transformer_2=self.get_module("transformer_2"),
                                            scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = WanV2VDepthPipeline
