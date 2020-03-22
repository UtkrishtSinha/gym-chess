import gym
from copy import deepcopy
from gym import error, spaces, utils
import numpy as np
from gym.utils import seeding


class ChessEnv(gym.Env):
    metadata = {'render.modes': ['human', 'pgn', 'none']}
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

    def __init__(self, player=None, state=None):
        self.player = player
        self.state = state
        self.empty = ['.', 'x', ' ', '', '0', 'o', 0, None]
        self.opponent = [-1 * self.player, -2 * self.player, -3 * self.player,
                         -4 * self.player, -5 * self.player, -6 * self.player]
        self.teammate = [1 * self.player, 2 * self.player, 3 * self.player,
                         4 * self.player, 5 * self.player, 6 * self.player]
        self.done = False
        self.move_string = None
        self.game_file = None
        self.observation_space = spaces.Box(-6, 6, (64, ), dtype=int)
        self.action_space = spaces.Discrete(n=100)
        # print(type(self.action_space))
        self.available_moves = []
        self.opponent_moves = []
        self.move_count = 0
        self.castling = True
        self.nn = None
        return

    def step(self, action):
        self.compute_moves()
        move_count = len(self.available_moves)
        action = int(str(action))
        action = action % move_count
        self.move_string = self.action_notation(ind=action)
        new_state = self.move(self.available_moves[action], simulate=False)
        reward = self.compute_reward(action)
        done = self.is_over()
        info = self.move_string
        if self.player == 1:
            self.move_count = self.move_count + 1
        return new_state, reward, done, info

    def reset(self):
        # state = [-3,   -5,  -4,  -2,  -1,  -4,  -5,  -3,
        #               -6,   -6,  -6,  -6,  -6,  -6,  -6,  -6,
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               6,     6,   6,   6,   6,   6,   6,   6,
        #               3,     5,   4,   2,   1,   4,   5,   3]
        state = ['.', '.', '.', '.', -1, '.', '.', '.',
                      '.', '.', '.', '.', '.', '.', '.', '.',
                      '.', '.', '.', '.', '.', '.', '.', '.',
                      '.', '.', '.', '.', '.', '.', '.', '.',
                      '.', '.', '.', '.', '.', '.', '.', '.',
                      '.', '.', '.', '.', '.', '.', '.', '.',
                      6, '.', '.', '.', '.', '.', '.', '.',
                      '.', '.', '.', '.', 1, '.', '.', '.']

        if self.state is None:
            self.state = []
            for i in range(64):
                self.state.append(state[i])

        else:
            for i in range(64):
                self.state[i] = state[i]
        self.done = False
        pass

    def render(self, mode='human'):
        if mode == 'human':
            self.print_state()
        if mode == 'pgn':
            if self.player == 1:
                self.game_file.write(str(self.move_count + 1) + "." + self.move_string + " ")
            else:
                self.game_file.write(self.move_string + "  ")
        if mode == 'none':
            pass

    def compute_reward(self, action):
        self.compute_moves()
        opp_moves = deepcopy(self.opponent_moves)
        self.eliminate_move(opponent=True)
        reward = 0
        for move in self.available_moves:
            reward = reward + ChessEnv.piece_values[self.state[move[0]]]
        for move in self.opponent_moves:
            reward = reward - ChessEnv.piece_values[self.state[move[0]]]
        if self.is_check(opponent=True) and len(self.opponent_moves) == 0:
            reward = reward + 10000
            self.move_string = self.move_string + "#"
        elif self.is_check(opponent=True):
            reward = reward + 100
            self.move_string = self.move_string + "+"
        self.opponent_moves = deepcopy(opp_moves)
        return reward

    def is_check(self, state=None, opponent=False):
        if state is None:
            state = self.state
        king = self.player
        if opponent:
            king = king * -1
        king_pos = None

        for ind in range(64):
            if state[ind] == king:
                king_pos = ind
                break
        king_row = king_pos // 8
        king_col = king_pos % 8
        if king == 1:
            opponent = [-2, -3]
        else:
            opponent = [2, 3]
        for row in range(king_row + 1, 8):
            pos = 8 * row + king_col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row in range(king_row - 1, -1, -1):
            pos = 8 * row + king_col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for col in range(king_col + 1, 8):
            pos = 8 * king_row + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for col in range(king_col - 1, -1, -1):
            pos = 8 * king_row + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        if king == 1:
            opponent = [-2, -4]
        else:
            opponent = [2, 4]
        for row_ul in range(king_row + 1, 8):
            col = (king_row + king_col) - row_ul
            if col > 7 or col < 0:
                break
            pos = 8 * row_ul + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row_ur in range(king_row + 1, 8):
            col = row_ur + (king_col - king_row)
            if col > 7 or col < 0:
                break
            pos = 8 * row_ur + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row_dr in range(king_row - 1, -1, -1):
            col = (king_row + king_col) - row_dr
            if col > 7 or col < 0:
                break
            pos = 8 * row_dr + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row_dl in range(king_row - 1, -1, -1):
            col = row_dl + (king_col - king_row)
            if col > 7 or col < 0:
                break
            pos = 8 * row_dl + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        if king == 1:
            if 0 <= king_row < 6 and 0 < king_col < 7:
                pos = 8 * (king_row + 1) + (king_col + 1)
                if state[pos] == -6:
                    return True
                pos = 8 * (king_row + 1) + (king_col - 1)
                if state[pos] == -6:
                    return True
            elif 0 <= king_row < 6 and king_col == 0:
                pos = 8 * (king_row + 1) + (king_col + 1)
                if state[pos] == -6:
                    return True
            elif 0 <= king_row < 6 and king_col == 7:
                pos = 8 * (king_row + 1) + (king_col - 1)
                if state[pos] == -6:
                    return True

        if king == -1:
            if 1 < king_row <= 7 and 0 < king_col < 7:
                pos = 8 * (king_row - 1) + (king_col + 1)
                if state[pos] == 6:
                    return True
                pos = 8 * (king_row - 1) + (king_col - 1)
                if state[pos] == 6:
                    return True
            elif 1 < king_row <= 7 and king_col == 0:
                pos = 8 * (king_row - 1) + (king_col + 1)
                if state[pos] == 6:
                    return True
            elif 1 < king_row <= 7 and king_col == 7:
                pos = 8 * (king_row - 1) + (king_col - 1)
                if state[pos] == 6:
                    return True

        if king_row + 2 < 8:
            if king_col + 1 < 8:
                pos = 8 * (king_row + 2) + (king_col + 1)
                if state[pos] == (-5 * king):
                    return True

            if king_col - 1 >= 0:
                pos = 8 * (king_row + 2) + (king_col - 1)
                if state[pos] == (-5 * king):
                    return True

        if king_row + 1 < 8:
            if king_col + 2 < 8:
                pos = 8 * (king_row + 1) + (king_col + 2)
                if state[pos] == (-5 * king):
                    return True

            if king_col - 2 >= 0:
                pos = 8 * (king_row + 1) + (king_col - 2)
                if state[pos] == (-5 * king):
                    return True

        if king_row - 1 >= 0:
            if king_col + 2 < 8:
                pos = 8 * (king_row - 1) + (king_col + 2)
                if state[pos] == (-5 * king):
                    return True

            if king_col - 2 >= 0:
                pos = 8 * (king_row - 1) + (king_col - 2)
                if state[pos] == (-5 * king):
                    return True

        if king_row - 2 >= 0:
            if king_col + 1 < 8:
                pos = 8 * (king_row - 2) + (king_col + 1)
                if state[pos] == (-5 * king):
                    return True

            if king_col - 1 >= 0:
                pos = 8 * (king_row - 2) + (king_col - 1)
                if state[pos] == (-5 * king):
                    return True

        return False

    def action_notation(self, ind):
        indic = 'o'
        # ind = self.action_space[action]
        pos_from = self.available_moves[ind][0]
        pos_to = self.available_moves[ind][1]
        promoted = self.available_moves[ind][2]
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
        # if move == '':
        #     move = str(chr(97 + (pos_from % 8)))
        if self.state[pos_to] in self.opponent:
            move = move + 'x'
            if self.state[pos_to] != 'x' and self.state[pos_to] != indic:
                move = move + ChessEnv.piece_notations[self.state[pos_to]]
        move = move + str(chr(97 + (pos_to % 8))) + str(9 - ((pos_to // 8) + 1))
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

    def set_game_file(self, file):
        self.game_file = file

    def print_state(self):
        for ind in range(8):
            for j in range(8):
                index = 8*ind + j
                print(self.state[index], end="\t")
            print()

    def set_nn(self, nn):
        self.nn = nn

    def compute_moves(self):
        indic = 'o'
        self.available_moves.clear()
        for ind in range(64):
            if self.state[ind] in ['.', ' ', 'x', 0]:
                continue
            if self.state[ind] in self.opponent:
                self.opponent_moves.extend(self.compute_move_for(ind, opponent=True))
                continue
            self.available_moves.extend(self.compute_move_for(ind))
        self.check_enable_castling()
        self.check_enable_castling(opponent=True)
        if self.player == 1:
            if self.state[2] == indic:
                self.opponent_moves.append((4, 2, None))
            if self.state[6] == indic:
                self.opponent_moves.append((4, 6, None))
            if self.state[58] == indic:
                self.available_moves.append((60, 58, None))
            if self.state[62] == indic:
                self.available_moves.append((60, 62, None))
        elif self.player == -1:
            if self.state[2] == indic:
                self.available_moves.append((4, 2, None))
            if self.state[6] == indic:
                self.available_moves.append((4, 6, None))
            if self.state[58] == indic:
                self.opponent_moves.append((60, 58, None))
            if self.state[62] == indic:
                self.opponent_moves.append((60, 62, None))
        self.eliminate_move()

    def compute_move_for(self, pos, opponent=False):
        moves = []
        if self.state[pos] in [6, -6]:
            moves = self.pawn_move(pos, opponent=opponent)
        elif self.state[pos] in [5, -5]:
            moves = self.knight_move(pos, opponent=opponent)
        elif self.state[pos] in [4, -4]:
            moves = self.bishop_move(pos, opponent=opponent)
        elif self.state[pos] in [3, -3]:
            moves = self.rook_move(pos, opponent=opponent)
        elif self.state[pos] in [2, -2]:
            moves = self.queen_move(pos, opponent=opponent)
        elif self.state[pos] in [1, -1]:
            moves = self.king_move(pos, opponent=opponent)
        return moves

    def move(self, move, simulate=True):
        if simulate:
            board = deepcopy(self.state)
        else:
            board = self.state
        piece = board[move[0]]
        board[move[0]] = '.'
        # case of en passant
        if board[move[1]] == 'x':
            if self.player == 1:
                board[move[1]] = piece
                board[move[1] + 8] = '.'
            else:
                board[move[1]] = piece
                board[move[1] - 8] = '.'

        # case of castling
        if piece == self.player and move[0] - move[1] == 2:
            board[move[1]] = piece
            board[move[1] - 2] = '.'
            board[move[1] + 1] = 3
        elif piece == self.player and move[0] - move[1] == -2:
            board[move[1]] = piece
            board[move[1] + 1] = '.'
            board[move[1] - 1] = 3

        # case of pawn promotion
            if move[2] not in ['.', 'x', 0, None, ' ', '']:
                board[move[1]] = move[2]

        # case of pawn promotion
        if move[2] is not None:
            board[move[1]] = move[2]

        self.clear_en_passant(board)
        if piece in [6, -6] and move[0] - move[1] in [16, -16]:
            board[(move[0] + move[1]) // 2] = 'x'

        # rest case
        if board[move[1]] in self.empty:
            board[move[1]] = piece

        return board

    def clear_en_passant(self, state=None):
        if state is None:
            state = self.state
        for ind in range(64):
            if state[ind] == 'x':
                state[ind] = '.'

    def add_move(self, move_list, move_from, move_to, promotion=None):
        move_to = 8 * move_to[0] + move_to[1]
        move_list.append((move_from, move_to, promotion))

    def pawn_move(self, pos, opponent=False):
        moves = []
        if not opponent:
            mate, opp = deepcopy(self.teammate), deepcopy(self.opponent)
        else:
            opp, mate = deepcopy(self.teammate), deepcopy(self.opponent)
        opp.append('x')
        row = pos // 8
        col = pos % 8
        if self.state[pos] == -6:
            if row < 6:
                if self.state[pos + 8] in self.empty:
                    self.add_move(moves, pos, (row + 1, col))
                    if row == 1 and self.state[pos + 16] in self.empty:
                        self.add_move(moves, pos, (row + 2, col))
                if col >= 1 and self.state[pos + 7] in opp:
                    self.add_move(moves, pos, (row + 1, col - 1))
                if col <= 6 and self.state[pos + 9] in opp:
                    self.add_move(moves, pos, (row + 1, col + 1))
            elif row == 6:
                if self.state[pos + 8] in self.empty:
                    self.add_move(moves, pos, (row + 1, col), -2)
                    self.add_move(moves, pos, (row + 1, col), -3)
                    self.add_move(moves, pos, (row + 1, col), -4)
                    self.add_move(moves, pos, (row + 1, col), -5)
                if col >= 1 and self.state[pos + 7] in opp:
                    self.add_move(moves, pos, (row + 1, col - 1), -2)
                    self.add_move(moves, pos, (row + 1, col - 1), -3)
                    self.add_move(moves, pos, (row + 1, col - 1), -4)
                    self.add_move(moves, pos, (row + 1, col - 1), -5)
                if col <= 6 and self.state[pos + 9] in opp:
                    self.add_move(moves, pos, (row + 1, col + 1), -2)
                    self.add_move(moves, pos, (row + 1, col + 1), -3)
                    self.add_move(moves, pos, (row + 1, col + 1), -4)
                    self.add_move(moves, pos, (row + 1, col + 1), -5)
        else:
            if row > 1:
                if self.state[pos - 8] in self.empty:
                    self.add_move(moves, pos, (row - 1, col))
                    if row == 6 and self.state[pos - 16] in self.empty:
                        self.add_move(moves, pos, (row - 2, col))
                if col >= 1 and self.state[pos - 9] in opp:
                    self.add_move(moves, pos, (row - 1, col - 1))
                if col <= 6 and self.state[pos - 7] in opp:
                    self.add_move(moves, pos, (row - 1, col + 1))
            elif row == 1:
                if self.state[pos - 8] in self.empty:
                    self.add_move(moves, pos, (row - 1, col), 2)
                    self.add_move(moves, pos, (row - 1, col), 3)
                    self.add_move(moves, pos, (row - 1, col), 4)
                    self.add_move(moves, pos, (row - 1, col), 5)
                if col >= 1 and self.state[pos - 9] in opp:
                    self.add_move(moves, pos, (row - 1, col - 1), 2)
                    self.add_move(moves, pos, (row - 1, col - 1), 3)
                    self.add_move(moves, pos, (row - 1, col - 1), 4)
                    self.add_move(moves, pos, (row - 1, col - 1), 5)
                if col <= 6 and self.state[pos - 7] in opp:
                    self.add_move(moves, pos, (row - 1, col + 1), 2)
                    self.add_move(moves, pos, (row - 1, col + 1), 3)
                    self.add_move(moves, pos, (row - 1, col + 1), 4)
                    self.add_move(moves, pos, (row - 1, col + 1), 5)
        return moves

    def king_move(self, pos, opponent=False):
        moves = []
        if not opponent:
            mate, opp = self.teammate, self.opponent
        else:
            opp, mate = self.teammate, self.opponent
        row = pos // 8
        col = pos % 8
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if not 0 <= r <= 7 or not 0 <= c <= 7:
                    continue
                p = 8 * r + c
                if self.state[p] in mate:
                    continue
                self.add_move(moves, pos, (r, c))
        return moves

    def bishop_move(self, pos, moves=None, opponent=False):
        if moves is None:
            moves = []
        if not opponent:
            mate, opp = self.teammate, self.opponent
        else:
            opp, mate = self.teammate, self.opponent
        row = pos // 8
        col = pos % 8
        s = row + col
        d = row - col
        # UL
        for r in range(row + 1, 8):
            c = s - r
            p = 8 * r + c
            if not 0 <= c <= 7:
                break
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (r, c))
                break
            else:
                self.add_move(moves, pos, (r, c))

        # UR
        for r in range(row + 1, 8):
            c = r - d
            p = 8 * r + c
            if not 0 <= c <= 7:
                break
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (r, c))
                break
            else:
                self.add_move(moves, pos, (r, c))

        # DR
        for r in range(row - 1, -1, -1):
            c = s - r
            p = 8 * r + c
            if not 0 <= c <= 7:
                break
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (r, c))
                break
            else:
                self.add_move(moves, pos, (r, c))

        # DL
        for r in range(row - 1, -1, -1):
            c = r - d
            p = 8 * r + c
            if not 0 <= c <= 7:
                break
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (r, c))
                break
            else:
                self.add_move(moves, pos, (r, c))

        return moves

    def rook_move(self, pos, moves=None, opponent=False):
        if moves is None:
            moves = []
        if not opponent:
            mate, opp = self.teammate, self.opponent
        else:
            opp, mate = self.teammate, self.opponent
        row = pos // 8
        col = pos % 8

        # Up
        for r in range(row + 1, 8):
            p = 8 * r + col
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (r, col))
                break
            else:
                self.add_move(moves, pos, (r, col))

        # Right
        for c in range(col + 1, 8):
            p = 8 * row + c
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (row, c))
                break
            else:
                self.add_move(moves, pos, (row, c))

        # Down
        for r in range(row - 1, -1, -1):
            p = 8 * r + col
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (r, col))
                break
            else:
                self.add_move(moves, pos, (r, col))

        # Left
        for c in range(col - 1, -1, -1):
            p = 8 * row + c
            if self.state[p] not in self.empty:
                if self.state[p] in mate:
                    break
                elif self.state[p] in opp:
                    self.add_move(moves, pos, (row, c))
                break
            else:
                self.add_move(moves, pos, (row, c))

        return moves

    def queen_move(self, pos, opponent=False):
        moves = []
        self.rook_move(pos, moves=moves, opponent=opponent)
        self.bishop_move(pos, moves=moves, opponent=opponent)
        return moves

    def knight_move(self, pos, opponent=False):
        moves = []
        if not opponent:
            mate, opp = self.teammate, self.opponent
        else:
            opp, mate = self.teammate, self.opponent
        row = pos // 8
        col = pos % 8
        if row + 2 < 8:
            if col + 1 < 8:
                p = 8 * (row + 2) + (col + 1)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row + 2, col + 1))

            if col - 1 >= 0:
                p = 8 * (row + 2) + (col - 1)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row + 2, col - 1))

        if row + 1 < 8:
            if col + 2 < 8:
                p = 8 * (row + 1) + (col + 2)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row + 2, col + 2))

            if col - 2 >= 0:
                p = 8 * (row + 1) + (col - 2)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row + 1, col - 2))

        if row - 1 >= 0:
            if col + 2 < 8:
                p = 8 * (row - 1) + (col + 2)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row - 1, col + 2))

            if col - 2 >= 0:
                p = 8 * (row - 1) + (col - 2)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row - 1, col - 2))

        if row - 2 >= 0:
            if col + 1 < 8:
                p = 8 * (row - 2) + (col + 1)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row - 2, col + 1))

            if col - 1 >= 0:
                p = 8 * (row - 2) + (col - 1)
                if self.state[p] not in mate:
                    self.add_move(moves, pos, (row - 2, col - 1))
        return moves

    def eliminate_move(self, opponent=False):
        if not opponent:
            i = 0
            while i < len(self.available_moves):
                board = self.move(self.available_moves[i], simulate=True)
                if self.is_check(board):
                    self.available_moves.pop(i)
                else:
                    i = i + 1
        else:
            i = 0
            while i < len(self.opponent_moves):
                board = self.move(self.opponent_moves[i], simulate=True)
                if self.is_check(board, opponent):
                    self.opponent_moves.pop(i)
                else:
                    i = i + 1

    def check_enable_castling(self, opponent=False):
        indic = 'o'
        state = self.state
        queen_castling = self.castling
        king_castling = self.castling
        opp_q_c = self.castling
        opp_k_c = self.castling
        if self.player == 1:
            if not opponent:
                if state[56] == 3 and state[60] == 1:
                    if state[58] in self.empty and state[59] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [58, 59]:
                                queen_castling = False
                                break
                        if queen_castling:
                            state[58] = indic
                if state[63] == 3 and state[60] == 1:
                    if state[61] in self.empty and state[62] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [62, 61]:
                                king_castling = False
                                break
                        if king_castling:
                            state[62] = indic
            else:
                if state[0] == -3 and state[4] == -1:
                    if state[2] in self.empty and state[3] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [2, 3]:
                                opp_q_c = False
                                break
                        if opp_q_c:
                            state[2] = indic
                if state[7] == -3 and state[4] == -1:
                    if state[5] in self.empty and state[6] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [5, 6]:
                                opp_k_c = False
                                break
                        if opp_k_c:
                            state[6] = indic
        elif self.player == -1:
            if not opponent:
                if state[0] == -3 and state[4] == -1:
                    if state[2] in self.empty and state[3] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [2, 3]:
                                queen_castling = False
                                break
                        if queen_castling:
                            state[2] = indic
                if state[7] == -3 and state[4] == 1:
                    if state[5] in self.empty and state[6] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [5, 6]:
                                king_castling = False
                                break
                        if king_castling:
                            state[6] = indic
            else:
                if state[56] == 3 and state[60] == 1:
                    if state[58] in self.empty and state[59] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [58, 59]:
                                opp_q_c = False
                                break
                        if opp_q_c:
                            state[58] = indic
                if state[63] == 3 and state[60] == 1:
                    if state[61] in self.empty and state[62] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [61, 62]:
                                opp_k_c = False
                                break
                        if opp_k_c:
                            state[62] = indic

    def is_over(self):
        opp_moves = deepcopy(self.opponent_moves)
        self.eliminate_move(opponent=True)
        if len(self.opponent_moves) == 0:
            self.opponent_moves = deepcopy(opp_moves)
            return True
        self.opponent_moves = deepcopy(opp_moves)
        return False

