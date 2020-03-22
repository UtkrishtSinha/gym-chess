from gym.envs.registration import register

register(
    id='chess-v0',
    entry_point='gym_chess.envs:ChessEnv',
)

register(
    id='ai_white-v0',
    entry_point='gym_chess.envs:ChessEnv',
    kwargs={'player': 1}
)

register(
    id='ai_black-v0',
    entry_point='gym_chess.envs:ChessEnv',
    kwargs={'player': -1}
)
