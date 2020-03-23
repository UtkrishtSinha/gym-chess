import gym
from copy import deepcopy
from gym import error, spaces, utils
import numpy as np
from gym.utils import seeding

def index_to_pos(index):
    file = str(chr(97 + (index % 8)))
    row = 8 - (index // 8)
    return file + str(row)


class MoveHandler:

    def __init__(self, p=None, state=None):
        self.player = 1
        self.opponent = []
        self.teammate = []
        self.empty = ['.', 'x', 'o']
        self.available_moves = []
        self.opponent_moves = []
        self.queen_castling = True
        self.king_castling = True
        self.opp_q_c = True
        self.opp_k_c = True
        self.state = None
        self.n = 0
        if p is not None:
            self.set_player(p)
        if state is not None:
            self.state = state
        else:
            self.state = ['.'] * 64

    def reset(self):
        if self.state is None:
            self.state = ['.'] * 64

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
        self.available_moves.clear()
        self.opponent_moves.clear()

    def set_state(self, state):
        for i in range(64):
            self.state[i] = state[i]

    def set_player(self, p):
        self.player = p
        self.teammate = [1 * p, 2 * p, 3 * p, 4 * p, 5 * p, 6 * p]
        self.opponent = [-1 * p, -2 * p, -3 * p, -4 * p, -5 * p, -6 * p]
        self.available_moves = []
        self.opponent_moves = []

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
        try:
            king_row = king_pos // 8
            king_col = king_pos % 8
        except TypeError:
            print(king)
            print(king_pos)
            self.print_state(state=state)
            print("----------------------------------------------------------------------------------------")
            self.render()
            print("action = ", self.n)
            exit(-3)
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
            if 0 < king_row <= 7 and 0 < king_col < 7:
                pos = 8 * (king_row - 1) + (king_col + 1)
                if state[pos] == -6:
                    return True
                pos = 8 * (king_row - 1) + (king_col - 1)
                if state[pos] == -6:
                    return True
            elif 0 < king_row <= 7 and king_col == 0:
                pos = 8 * (king_row - 1) + (king_col + 1)
                if state[pos] == -6:
                    return True
            elif 0 < king_row <= 7 and king_col == 7:
                pos = 8 * (king_row - 1) + (king_col - 1)
                if state[pos] == -6:
                    return True

        if king == -1:
            if 0 <= king_row < 7 and 0 < king_col < 7:
                pos = 8 * (king_row + 1) + (king_col + 1)
                if state[pos] == 6:
                    return True
                pos = 8 * (king_row + 1) + (king_col - 1)
                if state[pos] == 6:
                    return True
            elif 0 <= king_row < 7 and king_col == 0:
                pos = 8 * (king_row + 1) + (king_col + 1)
                if state[pos] == 6:
                    return True
            elif 0 <= king_row < 7 and king_col == 7:
                pos = 8 * (king_row + 1) + (king_col - 1)
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

        for row in range(king_row - 1, king_row + 2):
            for col in range(king_col - 1, king_col + 2):
                if not 0 <= row <= 7 or not 0 <= col <= 7:
                    continue
                p = 8 * row + col
                if state[p] == king * -1:
                    return True
        return False

    def compute_moves(self):
        indic = 'o'
        self.reset_castling()
        self.available_moves.clear()
        self.opponent_moves.clear()
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
        self.available_moves = self.eliminate_move()
        return self.available_moves, self.opponent_moves

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

        pos_from = move[0]
        pos_to = move[1]
        promotion = move[2]
        piece = board[pos_from]
        if not simulate:
            if piece in ['.', 'x', 'o']:
                print("Wrong move :- ", move, " for player = ", self.player)
                self.print_state(state=board)
                print("action = ", self.n)
                exit(-10)
            elif self.player * piece <= 0:
                print("Wrong move :- ", move, " for player = ", self.player)
                self.print_state(state=board)
                print("action = ", self.n)
                exit(-10)
        # Castling
        if piece in [1, -1] and (pos_from - pos_to) in [2, -2]:
            board[pos_from] = '.'
            board[pos_to] = piece
            board[(pos_from + pos_to) // 2] = 3 * self.player
            if pos_to % 8 == 2:
                board[pos_from - 4] = '.'
            else:
                board[pos_from + 3] = '.'
        # Pawn Promotion
        elif promotion in [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]:
            board[pos_from] = '.'
            board[pos_to] = promotion
        # En passant
        elif board[pos_to] == 'x' and piece in [-6, 6]:
            board[pos_from] = '.'
            board[pos_to] = piece
            if piece == 6:
                board[pos_to + 8] = '.'
            else:
                board[pos_to - 8] = '.'
        # Rest all case
        else:
            board[pos_from] = '.'
            board[pos_to] = piece
        if not simulate:
            if piece == self.player:
                self.king_castling = False
                self.queen_castling = False
            if piece == 3 * self.player and pos_from % 8 == 0:
                self.queen_castling = False
            if piece == 3 * self.player and pos_from % 8 == 7:
                self.king_castling = False
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
        row = pos // 8  # 6
        col = pos % 8  # 4
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
                    self.add_move(moves, pos, (row + 1, col + 2))

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
            return self.available_moves
        else:
            i = 0
            while i < len(self.opponent_moves):
                board = self.move(self.opponent_moves[i], simulate=True)
                if self.is_check(board, opponent):
                    self.opponent_moves.pop(i)
                else:
                    i = i + 1
            return self.opponent_moves

    def check_enable_castling(self, opponent=False):
        indic = 'o'
        state = self.state
        if self.player == 1:
            if not opponent:
                if state[56] == 3 and state[60] == 1:
                    if state[58] in self.empty and state[59] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [58, 59]:
                                self.queen_castling = False
                                break
                        if self.queen_castling:
                            state[58] = indic
                if state[63] == 3 and state[60] == 1:
                    if state[61] in self.empty and state[62] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [62, 61]:
                                self.king_castling = False
                                break
                        if self.king_castling:
                            state[62] = indic
            else:
                if state[0] == -3 and state[4] == -1:
                    if state[2] in self.empty and state[3] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [2, 3]:
                                self.opp_q_c = False
                                break
                        if self.opp_q_c:
                            state[2] = indic
                if state[7] == -3 and state[4] == -1:
                    if state[5] in self.empty and state[6] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [5, 6]:
                                self.opp_k_c = False
                                break
                        if self.opp_k_c:
                            state[6] = indic
        elif self.player == -1:
            if not opponent:
                if state[0] == -3 and state[4] == -1:
                    if state[2] in self.empty and state[3] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [2, 3]:
                                self.queen_castling = False
                                break
                        if self.queen_castling:
                            state[2] = indic
                if state[7] == -3 and state[4] == 1:
                    if state[5] in self.empty and state[6] in self.empty:
                        for m in self.opponent_moves:
                            if m[1] in [5, 6]:
                                self.king_castling = False
                                break
                        if self.king_castling:
                            state[6] = indic
            else:
                if state[56] == 3 and state[60] == 1:
                    if state[58] in self.empty and state[59] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [58, 59]:
                                self.opp_q_c = False
                                break
                        if self.opp_q_c:
                            state[58] = indic
                if state[63] == 3 and state[60] == 1:
                    if state[61] in self.empty and state[62] in self.empty:
                        for m in self.available_moves:
                            if m[1] in [61, 62]:
                                self.opp_k_c = False
                                break
                        if self.opp_k_c:
                            state[62] = indic

    def is_over(self):
        opp_moves = deepcopy(self.opponent_moves)
        self.opponent_moves = self.eliminate_move(opponent=True)
        if len(self.opponent_moves) == 0:
            self.opponent_moves = deepcopy(opp_moves)
            return True
        self.opponent_moves = deepcopy(opp_moves)
        return False

    def reset_castling(self, state=None):
        indic = 'o'
        if state is None:
            state = self.state
        for i in range(64):
            if state[i] == indic:
                state[i] = '.'

    def print_state(self, state=None):
        if state is None:
            state = self.state
        for ind in range(8):
            for j in range(8):
                index = 8*ind + j
                print(state[index], end="\t")
            print()


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

    def __init__(self, player=1, state=None):
        self.player = player
        self.state = state
        self.done = False
        self.move_string = None
        self.game_file = None
        self.move_file = None
        self.move_list = None
        self.observation_space = spaces.Box(-6, 6, (64, ), dtype=int)
        self.action_space = spaces.Discrete(n=100)
        # print(type(self.action_space))
        self.move_count = 0
        self.nn = None
        self.m = None
        self.n = -1
        self.move_handler = MoveHandler(p=self.player, state=self.state)
        return

    def step(self, action):
        if action == 12:
            a = 5
        self.compute_moves()
        move_count = len(self.move_handler.available_moves)
        action = int(str(action))
        action = action % move_count
        self.n = action
        self.move_handler.n = action
        self.move_string = self.action_notation(ind=action)
        self.m = self.move_handler.available_moves[action]
        new_state = self.move_handler.move(self.move_handler.available_moves[action], simulate=False)
        self.state = new_state
        reward = self.compute_reward(action)
        done = self.is_over()
        info = self.move_string
        self.move_count = self.move_count + 1
        self.move_handler.available_moves.clear()
        self.move_handler.opponent_moves.clear()
        return new_state, reward, done, info

    def reset(self):
        state = [-3,    -5,     -4,     -2,     -1,     -4,     -5,     -3,
                 -6,    -6,     -6,     -6,     -6,     -6,     -6,     -6,
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 '.',   '.',    '.',    '.',    '.',    '.',    '.',    '.',
                 6,     6,      6,      6,      6,      6,      6,      6,
                 3,     5,      4,      2,      1,      4,      5,      3]
        # state = ['.', '.', '.', '.', -1, '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', '.', '.', '.', '.',
        #               6, '.', '.', '.', '.', '.', '.', '.',
        #               '.', '.', '.', '.', 1, '.', '.', '.']

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

    def render(self, mode='human'):
        if mode == 'human':
            self.print_state()
        if mode == 'pgn':
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
        if mode == 'none':
            pass

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
        # ind = self.action_space[action]
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

    def is_over(self):
        opp_moves = deepcopy(self.move_handler.opponent_moves)
        self.move_handler.eliminate_move(opponent=True)
        if len(self.move_handler.opponent_moves) == 0:
            self.move_handler.opponent_moves = deepcopy(opp_moves)
            return True
        self.move_handler.opponent_moves = deepcopy(opp_moves)
        return False
