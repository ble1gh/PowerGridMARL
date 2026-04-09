#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from dataclasses import MISSING, dataclass


@dataclass
class TaskConfig:
    map_size: int = MISSING
    minimap_mode: bool = MISSING
    tag_penalty: float = MISSING
    max_cycles: int = MISSING
    extra_features: bool = MISSING
