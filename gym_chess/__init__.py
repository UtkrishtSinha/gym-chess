from gym.envs.registration import register

board = ['.'] * 64

register(
    id='ai_white-v0',
    entry_point='gym_chess.envs:ChessEnv',
    kwargs={'player': 1, 'state': board}
)

register(
    id='ai_black-v0',
    entry_point='gym_chess.envs:ChessEnv',
    kwargs={'player': -1, 'state': board}
)

register(
    id='Human-v0',
    entry_point='gym_chess.envs:PlayerEnv'
)

register(
    id='BotVsBot-v0',
    entry_point='gym_chess.envs:ChessGameTwoPlayers'
)

register(
    id='BotVsHuman-v0',
    entry_point='gym_chess.envs:ChessGameTwoPlayers',
    kwargs = {'bot_black': False}
)

register(
    id='HumanVsBot-v0',
    entry_point='gym_chess.envs:ChessGameTwoPlayers',
    kwargs = {'bot_white': False}
)

register(
    id='HumanVsHuman-v0',
    entry_point='gym_chess.envs:ChessGameTwoPlayers',
    kwargs = {'bot_white': False, 'bot_black': False}
)
