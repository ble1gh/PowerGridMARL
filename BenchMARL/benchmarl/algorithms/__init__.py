#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from .common import Algorithm, AlgorithmConfig
from .ensemble import EnsembleAlgorithm, EnsembleAlgorithmConfig
from .HGTeam import HGTeam, HGTeamConfig
from .HGTeamHA import HGTeamHAPPO, HGTeamHAPPOConfig
from .HGTeamSAC import HGTeamSAC, HGTeamSACConfig
from .iddpg import Iddpg, IddpgConfig
from .ippo import Ippo, IppoConfig
from .iql import Iql, IqlConfig
from .isac import Isac, IsacConfig
from .maddpg import Maddpg, MaddpgConfig
from .mappo import Mappo, MappoConfig
from .masac import Masac, MasacConfig
from .mat import MAT, MATConfig
from .qmix import Qmix, QmixConfig
from .vdn import Vdn, VdnConfig

classes = [
    "Iddpg",
    "IddpgConfig",
    "Ippo",
    "IppoConfig",
    "Iql",
    "IqlConfig",
    "Isac",
    "IsacConfig",
    "HGTeam",
    "HGTeamConfig",
    "HGTeamSAC",
    "HGTeamSACConfig",
    "HGTeamHAPPO",
    "HGTeamHAPPOConfig",
    "Maddpg",
    "MaddpgConfig",
    "Mappo",
    "MappoConfig",
    "MAT",
    "MATConfig",
    "Masac",
    "MasacConfig",
    "Qmix",
    "QmixConfig",
    "Vdn",
    "VdnConfig",
]

# A registry mapping "algoname" to its config dataclass
# This is used to aid loading of algorithms from yaml
algorithm_config_registry = {
    "mappo": MappoConfig,
    "mat": MATConfig,
    "ippo": IppoConfig,
    "maddpg": MaddpgConfig,
    "iddpg": IddpgConfig,
    "masac": MasacConfig,
    "isac": IsacConfig,
    "hgteam": HGTeamConfig,
    "hgteamsac": HGTeamSACConfig,
    "hgteamhappo": HGTeamHAPPOConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
    "iql": IqlConfig,
}
