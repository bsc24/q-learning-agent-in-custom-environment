from gym.envs.registration import register

register(
    id='BscMaze-v0',
    entry_point='maze_environment.envs:BscMazeEnvV0',
)

register(
    id='BscMaze-v1',
    entry_point='maze_environment.envs:BscMazeEnvV1',
)