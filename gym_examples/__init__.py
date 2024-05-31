from gymnasium.envs.registration import register

register(
    id="gym_examples/YFBasic",
    entry_point="gym_examples.envs:YFBasic",
    max_episode_steps=10000,
)

register(
    id="gym_examples/YFNormalized",
    entry_point="gym_examples.envs:YFNormalized",
    max_episode_steps=10000,
)

register(
    id="gym_examples/YFTechnical",
    entry_point="gym_examples.envs:YFTechnical",
    max_episode_steps=10000,
)

register(
    id="gym_examples/YFSharpe",
    entry_point="gym_examples.envs:YFSharpe",
    max_episode_steps=10000,
)

register(
    id="gym_examples/YF30",
    entry_point="gym_examples.envs:YF30",
    max_episode_steps=10000,
)
