from gym.envs.registration import register

register(
    id='BscMaze-v0',
    entry_point='maze_environment.envs:BscMazeEnvV0',
)
