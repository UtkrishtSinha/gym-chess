import gym
from gym_chess.envs.classes import MoveHandler, Sprite, Board, Player
from gym import spaces
from copy import deepcopy
from gym.envs.classic_control.rendering import Viewer
import re
import pyglet


def index_to_pos(index):
    file = str(chr(97 + (index % 8)))
    row = 8 - (index // 8)
    return file + str(row)


class ChessEnv(gym.Env):
    metadata = {'render.modes': ['human', 'state', 'board', 'pieces']}
    piece_values = [0, 1, 9, 5, 3, 3, 1, 1, 3, 3, 5, 9, 0]
    piece_notations = ['', 'K', 'Q', 'R', 'B', 'N', '', '', 'N', 'B', 'R', 'Q', 'K']

    # 6 = Pawn
    # 5 = Knight
    # 4 = Bishop
    # 3 = Rook
    # 2 = Queen
    # 1 = King
    # . = Empty Space
    # x = en passant
    # Positive means White
    # Negative means Black

    def __init__(self, player=1, state=None, board=None, board_width=512, board_height=512):
        self.player = player
        self.state = state
        self.done = False
        self.move_string = None
        self.game_file = None
        self.move_file = None
        self.move_list = None
        self.observation_space = spaces.Box(-6, 6, (64, ), dtype=int)
        self.action_space = spaces.Discrete(n=100)
        self.move_count = 0
        self.nn = None
        self.m = None
        self.n = -1
        self.move_handler = MoveHandler(p=self.player, state=self.state)
        self.move = None
        self.captured_piece = None
        self.en_p = None
        self.viewer = None
        self.game_board = board
        if board is None:
            self.game_board = Board(board_width, board_height, piece_arr)
        self.a = board_width
        self.b = board_height
        self.should_close = False
        self.m_x = 0
        self.m_y = 0
        return

    def step(self, action):
        self.compute_moves()
        move_count = len(self.move_handler.available_moves)
        if move_count == 0:
            return self.state, 0, True, {'tuple': None, 'notation':'', 'description': 'Finished'}
        action = int(str(action))
        action = action % move_count
        # self.move = self.move_handler.available_moves[action]
        self.n = action
        self.move_handler.n = action
        self.move_string = self.action_notation(ind=action)
        self.m = self.move_handler.available_moves[action]
        new_state, self.en_p = self.move_handler.move(self.m, action, simulate=False)
        # self.state = new_state
        reward = self.compute_reward(action)
        done = self.move_handler.is_over()
        self.game_board.update(self.m, True)
        self.move_count = self.move_count + 1
        self.move_handler.available_moves.clear()
        self.move_handler.opponent_moves.clear()
        return new_state, reward, done, {'tuple': self.m, 'notation': self.move_string, 'description': self.get_description()}

    def reset(self):
        state = [-3,    -5,     -4,     -2,     -1,     -4,     -5,     -3,
                 -6,    -6,     -6,     -6,     -6,     -6,     -6,     -6,
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 6,     6,      6,      6,      6,      6,      6,      6,
                 3,     5,      4,      2,      1,      4,      5,      3]

        if self.state is None:
            self.state = []
            for i in range(64):
                self.state.append(state[i])

        else:
            for i in range(64):
                self.state[i] = state[i]
        self.done = False
        self.move_count = 0
        self.move_handler.available_moves.clear()
        self.move_handler.opponent_moves.clear()
        self.move_handler.reset()
        self.game_board.reset()

    def render(self, mode='human', player=None, mouse_entered=None, mouse_exit=None, mouse_pressed=None):
        if mode == 'state':
            self.print_state()
            return
        if mode == 'human':
            if player is None:
                player = self.player
            if self.move_string is None:
                return
            n = None
            ml = None
            f = open(self.game_file, 'a')
            if self.move_file is not None:
                n = open(self.move_file, 'a')
            if self.move_list is not None:
                ml = open(self.move_list, 'a')
            if player == 1:
                if self.move_count == 1:
                    f.write("\n\n")
                f.write(str(self.move_count) + "." + self.move_string + " ")
                if n:
                    n.write(str(self.move_count) + "." + str(self.m) + "\n")
            else:
                if self.move_count % 10 == 0:
                    f.write(self.move_string + "\n")
                else:
                    f.write(self.move_string + "  ")
                if n:
                    n.write("..." + str(self.m) + "\n")
            if ml:
                if player == 1 and self.move_count == 1:
                    ml.write("\n\n")
                ml.write(str(self.n) + ",")
                if self.move_count % 10 == 0 and player == -1:
                    ml.write("\n         ")
                else:
                    ml.write(" ")
            f.close()
            if n:
                n.close()
            return
        if mode == 'board':
            if self.viewer is None and not self.should_close:
                from gym.envs.classic_control.rendering import Viewer
                self.viewer = Viewer(self.a, self.b)
                if mouse_entered:
                    self.unwrapped.viewer.window.on_mouse_enter = mouse_entered
                if mouse_exit:
                    self.unwrapped.viewer.window.on_mouse_leave = mouse_exit
                if mouse_pressed:
                    self.unwrapped.viewer.window.on_mouse_press = mouse_pressed
                self.unwrapped.viewer.window.on_close = self.close_window
                self.viewer.add_geom(self.game_board)
            if self.should_close:
                self.viewer.window.close()
                self.viewer = None
                return False
            return self.viewer.render(False)
        if mode == 'pieces':
            pieces = []
            for i in range(64):
                pieces.append(self.game_board.pieces[i])
            return pieces

    def set_mouse(self, x, y):
        self.m_x = x
        self.m_y = y

    def get_mouse(self):
        return self.m_x, self.m_y

    def compute_reward(self, action):
        self.compute_moves()
        opp_moves = deepcopy(self.move_handler.opponent_moves)
        self.move_handler.eliminate_move(opponent=True)
        reward = 0
        for move_i in self.move_handler.available_moves:
            try:
                reward = reward + ChessEnv.piece_values[self.state[move_i[0]]]
            except TypeError as e:
                print(move_i)
                self.render()
                exit(-1)
        for move in self.move_handler.opponent_moves:
            try:
                reward = reward - ChessEnv.piece_values[self.state[move[0]]]
            except TypeError:
                self.render()
                print(move)
                print("action = ", self.n)
                exit(-2)
        if self.move_handler.is_check(opponent=True) and len(self.move_handler.opponent_moves) == 0:
            reward = reward + 10000
            self.move_string = self.move_string + "#"
        elif self.move_handler.is_check(opponent=True):
            reward = reward + 100
            self.move_string = self.move_string + "+"
        self.move_handler.opponent_moves = deepcopy(opp_moves)
        return reward

    def action_notation(self, ind):
        indic = 'o'
        pos_from = self.move_handler.available_moves[ind][0]
        pos_to = self.move_handler.available_moves[ind][1]
        promoted = self.move_handler.available_moves[ind][2]
        if self.state[pos_to] == indic and self.state[pos_from] in [1, -1]:
            if pos_from - pos_to == 2:
                return "0-0-0"
            elif pos_from - pos_to == -2:
                return "0-0"
            else:
                return
        if self.state[pos_from] in ['.', 'x']:
            return
        move = ChessEnv.piece_notations[self.state[pos_from]]
        row_flag = False
        col_flag = False
        flag = False
        for m in self.move_handler.available_moves:
            if m[1] == pos_to and self.state[m[0]] == self.state[pos_from]:
                if m[0] == pos_from:
                    continue
                flag = True
                if row_flag:
                    pass
                else:
                    if m[0] % 8 == pos_from % 8:
                        row_flag = True

                if col_flag:
                    pass
                else:
                    if m[0] // 8 == pos_from // 8:
                        col_flag = True
        if row_flag and col_flag:
            move = move + index_to_pos(pos_from)
        elif row_flag and self.state[pos_from] not in [6, -6]:
            move = move + index_to_pos(pos_from)[1]
        elif col_flag or flag:
            move = move + index_to_pos(pos_from)[0]
        if move == '' and self.state[pos_to] in self.move_handler.opponent:
            move = index_to_pos(pos_from)[0]
        if self.state[pos_to] in self.move_handler.opponent:
            captured = [None, 'King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
            self.captured_piece = captured[self.state[pos_to]]
            move = move + 'x'
        move = move + index_to_pos(pos_to)
        if self.state[pos_from] in [1, -1]:
            if pos_from - pos_to == 2:
                move = "0-0-0"
            elif pos_from - pos_to == -2:
                move = "0-0"
        if promoted not in [0, '.', 'x', '', None]:
            move = move + '=' + ChessEnv.piece_notations[promoted]
        return move

    def set_state(self, state=None):
        if state is None:
            self.reset()
        else:
            self.state = state
        self.move_handler.set_state(self.state)

    def set_game_file(self, file, move_file=None, move_list=None):
        self.game_file = file
        self.move_file = move_file
        self.move_list = move_list

    def print_state(self, state=None):
        if state is None:
            state = self.state
        for ind in range(8):
            for j in range(8):
                index = 8*ind + j
                print(state[index], end="\t")
            print()

    def set_nn(self, nn):
        self.nn = nn

    def compute_moves(self):
        self.move_handler.compute_moves()

    def return_possible_moves(self, player=1):
        # self.player = player
        # self.move_handler.set_player(player)
        self.compute_moves()
        return self.move_handler.available_moves

    def get_description(self):
        return self.move_handler.get_description(string=self.move_string)

    def get_piece_by_code(self, code):
        pieces = ['King', 'Queen', 'Rook', 'Bishop', 'Knight']
        codes = ['K', 'Q', 'R', 'B', 'N']
        for i in range(len(codes)):
            if code == codes[i]:
                return pieces[i]
        return 'Pawn'

    def is_bot(self):
        return True

    def close(self):
        if self.viewer:
            self.viewer.window.close()
            self.viewer = None

    def close_window(self):
        self.should_close = True

    def __del__(self):
        self.close()

class PlayerEnv(gym.Env):
    metadata = {'render.modes': ['human', 'state', 'board']}

    def __init__(self, player, board=None, state=None):
        self.board = board
        if not board:
            self.board = Board(512, 512)
        self.p = player
        self.state = state
        if not state:
            self.state = ['.'] * 64
        self.observation_space = spaces.Box(-6, 6, (64,), dtype=int)
        self.action_space = spaces.Discrete(n=64)
        self.player = Player(player=self.p, state=self.state)
        self.move_handler = self.player.move_handler
        self.viewer = None
        self.fr = None
        self.to = None
        self.game_file = None
        self.move_file = None
        self.move_list = None
        self.move_string = None

    def step(self, action):
        move = self.move_handler.available_moves[action]
        self.move_string = self.move_handler.to_string(move)
        state, _ = self.move_handler.move(move=move, action=action, simulate=False)
        self.board.update(move, True)
        done = self.move_handler.is_over()
        desc = self.move_handler.get_description(string=self.move_string)
        return state, 0, done, {'tuple': move, 'notation': self.move_string, 'description': desc}

    def reset(self):
        self.player.move_handler.reset()

    def render(self, mode='human'):
        def render(self, mode='human', mouse_entered=None, mouse_exit=None, mouse_pressed=None):
            if mode == 'state':
                self.print_state()
                return
            if mode == 'human':
                n = None
                ml = None
                f = open(self.game_file, 'a')
                if self.move_file is not None:
                    n = open(self.move_file, 'a')
                if self.move_list is not None:
                    ml = open(self.move_list, 'a')
                if self.player == 1:
                    if self.move_count == 1:
                        f.write("\n\n")
                    f.write(str(self.move_count) + "." + self.move_string + " ")
                    if n:
                        n.write(str(self.move_count) + "." + str(self.m) + "\n")
                else:
                    if self.move_count % 10 == 0:
                        f.write(self.move_string + "\n")
                    else:
                        f.write(self.move_string + "  ")
                    if n:
                        n.write("..." + str(self.m) + "\n")
                if ml:
                    if self.player == 1 and self.move_count == 1:
                        ml.write("\n\n")
                    ml.write(str(self.n) + ",")
                    if self.move_count % 10 == 0 and self.player == -1:
                        ml.write("\n         ")
                    else:
                        ml.write(" ")
                f.close()
                if n:
                    n.close()
                return
            if mode == 'board':
                if self.viewer is None and not self.should_close:
                    from gym.envs.classic_control.rendering import Viewer
                    self.viewer = Viewer(self.a, self.b)
                    if mouse_entered:
                        self.unwrapped.viewer.window.on_mouse_enter = mouse_entered
                    if mouse_exit:
                        self.unwrapped.viewer.window.on_mouse_leave = mouse_exit
                    if mouse_pressed:
                        self.unwrapped.viewer.window.on_mouse_press = mouse_pressed
                    self.unwrapped.viewer.window.on_close = self.close_window
                    self.viewer.add_geom(self.game_board)
                if self.should_close:
                    self.viewer.window.close()
                    self.viewer = None
                    return False
                return self.viewer.render(False)
            if mode == 'pieces':
                pieces = []
                for i in range(64):
                    pieces.append(self.game_board.pieces[i])
                return pieces

    def set_game_file(self, file, move_file=None, move_list=None):
        self.game_file = file
        self.move_file = move_file
        self.move_list = move_list

    def is_bot(self):
        return False

    def compute_moves(self):
        self.move_handler.compute_moves()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def __del__(self):
        self.close()

class ChessGameTwoPlayers(gym.Env):
    metadata = {'render.modes': ['human', 'state', 'board']}

    def __init__(self, board_w=512, board_h=512, bot_white=True, bot_black=True):
        self.state = ['.'] * 64
        self.board = Board(width=board_w, height=board_h)
        self.a = board_w
        self.b = board_h
        self.observation_space = spaces.Box(-6, 6, (64,), dtype=int)
        self.action_space = spaces.Discrete(n=100)
        if bot_white:
            self.player1 = ChessEnv(player=1, state=self.state, board=self.board,
                                board_width=board_w, board_height=board_h)
        else:
            self.player1 = PlayerEnv(player=1, state=self.state, board=self.board)

        if bot_black:
            self.player2 = ChessEnv(player=-1, state=self.state, board=self.board,
                                board_width=board_w, board_height=board_h)
        else:
            self.player2 = PlayerEnv(player=-1, state=self.state, board=self.board)

        self.playing = self.player1
        self.viewer = None
        self.should_close = False
        self.mouse_on = True
        self.pos = None
        self.fr = None
        self.to = None
        self.played = None
        self.promotion = False
        self.promoted = None

    def step(self, action):
        if self.playing.is_bot():
            state, reward, done, info = self.playing.step(action)
            self.played = self.playing
            self.switch_player()
            return state, reward, done, info
        else:
            move = None
            if self.pos is not None or self.promoted is not None:
                move = self.get_move(self.pos)
                self.board.clear_highlight()
                if move:
                    self.board.clear_prom()
                    action = self.move_to_action(move)
                    state, reward, done, info = self.playing.step(action)
                    self.switch_player()
                    self.promoted = None
                    self.promotion = False
                    self.pos = None
                    return state, reward, done, info
                else:
                    self.highlight_moves(int(self.pos))
                self.pos = None
            state = self.playing.state
            reward = 0
            done = self.playing.move_handler.is_over()
            notation = self.playing.move_handler.to_string(move)
            desc = self.playing.move_handler.get_description(move=move)
            info = {'tuple': move, 'notation': notation, 'description': desc}
            return state, reward, done, info

    def reset(self):
        state = [-3, -5, -4, -2, -1, -4, -5, -3,
                 -6, -6, -6, -6, -6, -6, -6, -6,
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 6, 6, 6, 6, 6, 6, 6, 6,
                 3, 5, 4, 2, 1, 4, 5, 3]
        for i in range(64):
            self.state[i] = state[i]
        self.board.reset()
        self.playing = self.player1
        self.playing.compute_moves()

    def render(self, mode='board',  mouse_entered=None, mouse_exit=None, mouse_pressed=None):
        if mode=='human':
            if self.played:
                self.played.render(mode)
            return True
        elif mode=='state':
            if self.played:
                self.played.render(mode)
            return True
        elif mode=='board':
            if self.should_close:
                return False
            if not self.viewer:
                self.viewer = Viewer(self.a + self.a//8 + 32, self.b)
                if mouse_entered:
                    self.unwrapped.viewer.window.on_mouse_enter = mouse_entered
                else:
                    self.unwrapped.viewer.window.on_mouse_enter = self.mouse_enter
                if mouse_exit:
                    self.unwrapped.viewer.window.on_mouse_leave = mouse_exit
                else:
                    self.unwrapped.viewer.window.on_mouse_leave = self.mouse_exit
                if mouse_pressed:
                    self.unwrapped.viewer.window.on_mouse_press = mouse_pressed
                else:
                    self.unwrapped.viewer.window.on_mouse_press = self.mouse
                self.unwrapped.viewer.window.on_close = self.close_window
                self.viewer.add_geom(self.board)
            try:
                self.viewer.render(False)
            except AttributeError as e:
                print(e)
            return True
        elif mode=='pieces':
            pieces = []
            for i in range(64):
                pieces.append(self.board.pieces[i])
            return pieces

    def switch_player(self):
        if self.playing == self.player1:
            self.playing = self.player2
        else:
            self.playing = self.player1
        self.playing.compute_moves()

    def set_game_file(self, file, move_file=None, move_list=None):
        self.player1.set_game_file(file, move_file, move_list)
        self.player2.set_game_file(file, move_file, move_list)

    def print_state(self):
        self.playing.print_state()

    def close_window(self):
        if self.prom_view:
            self.prom_view.close()
            self.prom_view = None

        if self.viewer:
            self.viewer.close()
            self.viewer = None
            self.should_close = True

    def mouse(self, x, y, button, modifiers):
        if self.playing.is_bot():
            self.pos = None
            return
        if button == pyglet.window.mouse.LEFT:
            self.pos = self.mouse_to_pos(x, y)
            self.promoted = self.get_promoted(x, y)

    def get_promoted(self, x, y):
        if x < self.a + 32:
            return None
        if y < self.b // 8:
            return -2
        elif y < 2 * self.b // 8:
            return -3
        elif y < 3 * self.b // 8:
            return -4
        elif y < 4 * self.b // 8:
            return -5
        elif y < 5 * self.b // 8:
            return 5
        elif y < 6 * self.b // 8:
            return 4
        elif y < 7 * self.b // 8:
            return 3
        elif y < self.b:
            return 2
        else:
            return None

    def mouse_enter(self, x, y):
        self.mouse_on = True

    def mouse_exit(self, x, y):
        self.mouse_on = False

    def mouse_to_pos(self, x, y):
        if x is None or y is None:
            return None
        if self.promotion:
            return None
        r = 7 - (y // (self.b/8))
        c = x // (self.a/8)
        if 0 <= r <= 7 and 0 <= c <= 7:
            return 8 * r + c
        else:
            None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def highlight_moves(self, pos):
        if self.state[pos] not in self.playing.move_handler.teammate:
            return
        row = pos // 8
        col = pos % 8

        if row % 2 == col % 2:
            selected = Sprite('Textures/Selected_Light.png', self.a/8, self.b/8)
        else:
            selected = Sprite('Textures/Selected_Dark.png', self.a/8, self.b/8)

        self.board.set_highlight(pos, selected)

        for m in self.playing.move_handler.available_moves:
            if m[0] == pos:
                moves = Sprite('Textures/moves.png', self.a / 8, self.b / 8)
                self.board.set_highlight(m[1], moves)

    def get_move(self, pos):
        if pos is not None:
            pos = int(pos)
        if not self.promotion:
            if self.state[pos] in self.playing.move_handler.teammate:
                self.fr = pos
            if not self.fr:
                return None
            else:
                for move in self.playing.move_handler.available_moves:
                    if move[0] == self.fr and move[1] == pos:
                        self.to = pos
                        if move[2]:
                            self.promotion = True
                            self.board.create_prom_menu(self.playing.p)
                            return None
                        else:
                            return move
        else:
            if self.promoted:
                return (self.fr, self.to, self.promoted)
            else:
                return None

    def move_to_action(self, move):
        for i in range(len(self.playing.move_handler.available_moves)):
            if self.playing.move_handler.available_moves[i] == move:
                return i

    def close_prom(self):
        print("closing...")

    def __del__(self):
        self.close()
