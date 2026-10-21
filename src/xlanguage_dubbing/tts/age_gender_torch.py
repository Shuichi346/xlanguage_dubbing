"""Load audEERING's two prediction heads without executing remote model code."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    """Preserve the checkpoint's dense/tanh/output projection architecture."""

    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        features = self.dropout(features)
        features = torch.tanh(self.dense(features))
        return self.out_proj(self.dropout(features))


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Match the provider's custom age and gender heads, which AutoModel omits."""

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0].mean(dim=1)
        # Return raw gender logits; profile_from_outputs applies softmax once.
        return self.age(hidden), self.gender(hidden)


class HuggingFacePredictor:
    """Present the same raw-output contract as the local ONNX backend."""

    def __init__(self, root: Path):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(str(root), local_files_only=True)
        # Transformers' annotation omits the tuple returned with loading info.
        model, loading = cast(
            tuple[AgeGenderModel, dict],
            AgeGenderModel.from_pretrained(
                str(root),
                local_files_only=True,
                use_safetensors=True,
                dtype=torch.float32,
                output_loading_info=True,
            ),
        )
        if (
            loading.get("missing_keys")
            or loading.get("mismatched_keys")
            or loading.get("error_msgs")
        ):
            raise ValueError(f"Incomplete age/gender checkpoint: {loading}")
        self.model = model.cpu().eval()

    def __call__(self, signal: np.ndarray, sample_rate: int) -> dict:
        inputs = self.processor(signal, sampling_rate=sample_rate, return_tensors="pt")
        values = inputs["input_values"].to(device="cpu", dtype=torch.float32)
        with torch.inference_mode():
            age, gender = self.model(values)
        return {"logits_age": age.numpy(), "logits_gender": gender.numpy()}
