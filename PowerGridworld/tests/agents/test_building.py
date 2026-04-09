from gridworld.agents.buildings import FiveZoneROMThermalEnergyEnv

from tests.conftest import single_agent_episode_runner


def test_default_env(building_config):
    """Creates an env instance and runs it (sanity check)."""
    env = FiveZoneROMThermalEnergyEnv(**building_config)
    assert single_agent_episode_runner(env)
