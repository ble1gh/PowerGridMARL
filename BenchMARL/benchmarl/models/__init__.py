#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .cnn import Cnn, CnnConfig
from .common import (
    EnsembleModelConfig,
    Model,
    ModelConfig,
    SequenceModel,
    SequenceModelConfig,
)
from .deepsets import Deepsets, DeepsetsConfig
from .edgebiasedHGT import EdgeBiasedHGT, EdgeBiasedHGTConfig
from .edgeweightedHGT import EdgeWeightedHGT, EdgeWeightedHGTConfig
from .gnn import Gnn, GnnConfig
from .gru import Gru, GruConfig
from .heterognn import HeteroGNN, HeteroGnnConfig
from .lstm import Lstm, LstmConfig
from .mlp import Mlp, MlpConfig
from .transformer import Transformer, TransformerConfig

classes = [
    "Mlp",
    "MlpConfig",
    "Gnn",
    "GnnConfig",
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "Gru",
    "GruConfig",
    "Lstm",
    "LstmConfig",
    "Transformer",
    "TransformerConfig",
    "EdgeBiasedHGT",
    "EdgeBiasedHGTConfig",
    "EdgeWeightedHGT",
    "EdgeWeightedHGTConfig",
]

__all__ = [
    "Cnn",
    "CnnConfig",
    "Deepsets",
    "DeepsetsConfig",
    "EdgeBiasedHGT",
    "EdgeBiasedHGTConfig",
    "EdgeWeightedHGT",
    "EdgeWeightedHGTConfig",
    "EnsembleModelConfig",
    "Gnn",
    "GnnConfig",
    "Gru",
    "GruConfig",
    "HeteroGNN",
    "HeteroGnnConfig",
    "Lstm",
    "LstmConfig",
    "Mlp",
    "MlpConfig",
    "Model",
    "ModelConfig",
    "SequenceModel",
    "SequenceModelConfig",
    "Transformer",
    "TransformerConfig",
]

model_config_registry = {
    "mlp": MlpConfig,
    "gnn": GnnConfig,
    "cnn": CnnConfig,
    "deepsets": DeepsetsConfig,
    "gru": GruConfig,
    "lstm": LstmConfig,
    "transformer": TransformerConfig,
    "edgebiasedhgt": EdgeBiasedHGTConfig,
    "edgeweightedhgt": EdgeWeightedHGTConfig,
}
