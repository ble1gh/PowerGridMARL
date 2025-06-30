#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import MappoConfig, MasacConfig, IppoConfig, MaddpgConfig, IddpgConfig, IsacConfig
from benchmarl.benchmark import Benchmark
from benchmarl.environments import PowerGridworldTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.models.mlp import MlpConfig

if __name__ == "__main__":

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml()

    # Loads from "benchmarl/conf/task/vmas"
    tasks = [PowerGridworldTask.EVOVERNIGHT13NODE.get_from_yaml()]

    # Loads from "benchmarl/conf/algorithm"
    algorithm_configs = [
        MappoConfig.get_from_yaml(),
        IppoConfig.get_from_yaml(),
        MasacConfig.get_from_yaml(),
        MaddpgConfig.get_from_yaml(),
        IddpgConfig.get_from_yaml(),
        IsacConfig.get_from_yaml(),
    ]

    # Loads from "benchmarl/conf/model/layers"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        tasks=tasks,
        experiment_config=experiment_config,
        seeds=[1, ],
        model_config=model_config,
        critic_model_config=critic_model_config,
    )
    benchmark.run_sequential()
