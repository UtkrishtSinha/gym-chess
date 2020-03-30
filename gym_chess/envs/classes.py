import pyglet
from gym.envs.classic_control.rendering import Geom
from copy import deepcopy
import re

def index_to_pos(index):
    file = str(chr(97 + (index % 8)))
    row = 8 - (index // 8)
    return file + str(row)


class MoveHandler:
    hash_values = [313, 317, 337, 347, 349, 353, 359, 367,
                   373, 379, 383, 389, 397, 401, 409, 419,
                   421, 431, 439, 443, 449, 457, 461, 463,
                   467, 479, 487, 491, 499, 503, 509, 521,
                   523, 541, 547, 557, 563, 569, 571, 577,
                   587, 593, 599, 601, 607, 613, 617, 619,
                   631, 641, 643, 647, 653, 659, 661, 673,
                   677, 683, 691, 701, 709, 719, 727, 733]

    piece_hash = [0, 7907, 7351, 6829, 5783, 5323, 4943, 4937, 5333, 5791, 6833, 7919, 7369]

    board_hashes = []
    no_pawn_count = 0

    def __init__(self, p=None, state=None):
        self.player = 1
        self.action = None
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
        self.en_p = None
        self.captured_piece = None

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
        if self.player == 1:
            MoveHandler.board_hashes.clear()
            MoveHandler.board_hashes.append(self.compute_hash())
        MoveHandler.no_pawn_count = 0

    def compute_hash(self, state=None, playerturn=None):
        if state is None:
            state = self.state
        if playerturn is None:
            playerturn = self.player
        hash = 0
        for i in range(64):
            if state[i] in ['.', 'x', 'o']:
                continue
            else:
                hash = hash + MoveHandler.hash_values[i] * MoveHandler.piece_hash[state[i]]

        return hash * playerturn

    def compute_hash_for(self, move):
        board = self.move(move, simulate=True)
        return self.compute_hash(board, self.player * - 1)

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

        if king_pos is None:
            exit(1000)
        else:
            return self.is_square_attacked(king_pos, state, opponent)

    def is_square_attacked(self, square, state=None, opponent=False):
        if state is None:
            state = self.state
        player = self.player
        if opponent:
            player = player * -1

        square_row = square // 8
        square_col = square % 8

        if player == 1:
            opponent = [-2, -3]
        else:
            opponent = [2, 3]
        for row in range(square_row + 1, 8):
            pos = 8 * row + square_col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row in range(square_row - 1, -1, -1):
            pos = 8 * row + square_col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for col in range(square_col + 1, 8):
            pos = 8 * square_row + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for col in range(square_col - 1, -1, -1):
            pos = 8 * square_row + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        if player == 1:
            opponent = [-2, -4]
        else:
            opponent = [2, 4]
        for row_ul in range(square_row + 1, 8):
            col = (square_row + square_col) - row_ul
            if col > 7 or col < 0:
                break
            pos = 8 * row_ul + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row_ur in range(square_row + 1, 8):
            col = row_ur + (square_col - square_row)
            if col > 7 or col < 0:
                break
            pos = 8 * row_ur + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row_dr in range(square_row - 1, -1, -1):
            col = (square_row + square_col) - row_dr
            if col > 7 or col < 0:
                break
            pos = 8 * row_dr + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        for row_dl in range(square_row - 1, -1, -1):
            col = row_dl + (square_col - square_row)
            if col > 7 or col < 0:
                break
            pos = 8 * row_dl + col
            if state[pos] in ['.', 'x']:
                continue
            if state[pos] in opponent:
                return True
            else:
                break

        if player == 1:
            if 0 < square_row <= 7 and 0 < square_col < 7:
                pos = 8 * (square_row - 1) + (square_col + 1)
                if state[pos] == -6:
                    return True
                pos = 8 * (square_row - 1) + (square_col - 1)
                if state[pos] == -6:
                    return True
            elif 0 < square_row <= 7 and square_col == 0:
                pos = 8 * (square_row - 1) + (square_col + 1)
                if state[pos] == -6:
                    return True
            elif 0 < square_row <= 7 and square_col == 7:
                pos = 8 * (square_row - 1) + (square_col - 1)
                if state[pos] == -6:
                    return True

        if player == -1:
            if 0 <= square_row < 7 and 0 < square_col < 7:
                pos = 8 * (square_row + 1) + (square_col + 1)
                if state[pos] == 6:
                    return True
                pos = 8 * (square_row + 1) + (square_col - 1)
                if state[pos] == 6:
                    return True
            elif 0 <= square_row < 7 and square_col == 0:
                pos = 8 * (square_row + 1) + (square_col + 1)
                if state[pos] == 6:
                    return True
            elif 0 <= square_row < 7 and square_col == 7:
                pos = 8 * (square_row + 1) + (square_col - 1)
                if state[pos] == 6:
                    return True

        if square_row + 2 < 8:
            if square_col + 1 < 8:
                pos = 8 * (square_row + 2) + (square_col + 1)
                if state[pos] == (-5 * player):
                    return True

            if square_col - 1 >= 0:
                pos = 8 * (square_row + 2) + (square_col - 1)
                if state[pos] == (-5 * player):
                    return True

        if square_row + 1 < 8:
            if square_col + 2 < 8:
                pos = 8 * (square_row + 1) + (square_col + 2)
                if state[pos] == (-5 * player):
                    return True

            if square_col - 2 >= 0:
                pos = 8 * (square_row + 1) + (square_col - 2)
                if state[pos] == (-5 * player):
                    return True

        if square_row - 1 >= 0:
            if square_col + 2 < 8:
                pos = 8 * (square_row - 1) + (square_col + 2)
                if state[pos] == (-5 * player):
                    return True

            if square_col - 2 >= 0:
                pos = 8 * (square_row - 1) + (square_col - 2)
                if state[pos] == (-5 * player):
                    return True

        if square_row - 2 >= 0:
            if square_col + 1 < 8:
                pos = 8 * (square_row - 2) + (square_col + 1)
                if state[pos] == (-5 * player):
                    return True

            if square_col - 1 >= 0:
                pos = 8 * (square_row - 2) + (square_col - 1)
                if state[pos] == (-5 * player):
                    return True

        for row in range(square_row - 1, square_row + 2):
            for col in range(square_col - 1, square_col + 2):
                if not 0 <= row <= 7 or not 0 <= col <= 7:
                    continue
                p = 8 * row + col
                if state[p] == player * -1:
                    return True
        return False

    def evaluate_board(self, player=None, board=None):
        return 1

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

    def move(self, move=None, action=None, simulate=True):
        if action:
            self.action = action
        self.en_p = None
        if simulate:
            board = deepcopy(self.state)
        else:
            board = self.state

        pos_from = move[0]
        pos_to = move[1]
        promotion = move[2]
        piece = board[pos_from]
        if not simulate:
            self.captured_piece = board[pos_to]
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
                self.en_p = pos_to + 8
            else:
                board[pos_to - 8] = '.'
                self.en_p = pos_to - 8
        # Rest all case
        else:
            board[pos_from] = '.'
            board[pos_to] = piece

        self.clear_en_passant(board)
        if piece in [-6, 6] and pos_from - pos_to in [-16, 16]:
            board[(pos_from + pos_to) // 2] = 'x'

        if not simulate:
            if piece == self.player:
                self.king_castling = False
                self.queen_castling = False
            if piece == 3 * self.player and pos_from % 8 == 0:
                self.queen_castling = False
            if piece == 3 * self.player and pos_from % 8 == 7:
                self.king_castling = False
            hash = self.compute_hash(board, self.player * -1)
            MoveHandler.board_hashes.append(hash)

            if piece == 6 * self.player:
                MoveHandler.no_pawn_count = 0
            else:
                MoveHandler.no_pawn_count += 1

        return board, self.en_p

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
                board, _ = self.move(self.available_moves[i], simulate=True)
                if self.is_check(board):
                    self.available_moves.pop(i)
                else:
                    i = i + 1
            return self.available_moves
        else:
            i = 0
            while i < len(self.opponent_moves):
                board, _ = self.move(self.opponent_moves[i], simulate=True)
                if self.is_check(board, opponent):
                    self.opponent_moves.pop(i)
                else:
                    i = i + 1
            return self.opponent_moves

    def get_piece_by_code(self, code):
        pieces = ['King', 'Queen', 'Rook', 'Bishop', 'Knight']
        codes = ['K', 'Q', 'R', 'B', 'N']
        for i in range(len(codes)):
            if code == codes[i]:
                return pieces[i]
        return 'Pawn'

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
                if state[7] == -3 and state[4] == -1:
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
        total_hashes = len(MoveHandler.board_hashes)
        hash = MoveHandler.board_hashes[total_hashes - 1]
        if MoveHandler.board_hashes.count(hash) >= 3:
            return True
        if MoveHandler.no_pawn_count >= 50:
            return True
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

    def to_string(self, move=None):
        if move is None:
            return ''
        piece_notations = ['', 'K', 'Q', 'R', 'B', 'N', '', '', 'N', 'B', 'R', 'Q', 'K']
        indic = 'o'
        pos_from = move[0]
        pos_to = move[1]
        promoted = move[2]
        if self.state[pos_to] == indic and self.state[pos_from] in [1, -1]:
            if pos_from - pos_to == 2:
                return "0-0-0"
            elif pos_from - pos_to == -2:
                return "0-0"
            else:
                return
        if self.state[pos_from] in ['.', 'x']:
            return
        move = piece_notations[self.state[pos_from]]
        row_flag = False
        col_flag = False
        flag = False
        for m in self.available_moves:
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
        if move == '' and self.state[pos_to] in self.opponent:
            move = index_to_pos(pos_from)[0]
        if self.state[pos_to] in self.opponent:
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
            move = move + '=' + piece_notations[promoted]
        return move

    def get_description(self, move=None, string=None):
        if move is None and string is None:
            return ''
        elif string:
            move_string = string
        else:
            move_string = self.to_string(move)
        players = [None, 'White', 'Black']
        player = players[self.player]
        code = ''
        if self.captured_piece in [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6]:
            captured = [None, 'King', 'Queen', 'Rook', 'Bishop', 'Knight', 'Pawn', 'Knight', 'Bishop', 'Rook', 'Queen',
                        'King']
            self.captured_piece = captured[self.captured_piece]
        if move_string[0] in ['K', 'Q', 'R', 'B', 'N']:
            code = move_string[0]
        piece = self.get_piece_by_code(code)
        if move_string is None:
            return None
        else:
            desc = None
            if move_string == '0-0-0':
                return player + ' castles Queen-Side'
            elif move_string == '0-0':
                return player + ' castles King-Side'
            elif 'x' in move_string:
                pos = re.findall(r'x[a-h][1-8]', move_string)[0]
                try:
                    desc = player + ' captures ' + self.captured_piece + ' at ' + pos[1] + pos[2] + ' using ' + piece
                except TypeError:
                    print(move_string)
                    print(player)
                    print(self.captured_piece)
                    print(pos)
                    print(piece)
                    print('action = ', self.action)
                    exit(100)
            elif self.en_p:
                at = re.findall('[a-h][1-8]', move_string)
                at = at[len(at) - 1]
                pos = index_to_pos(self.en_p)
                desc =  player + ' en passants pawn at ' + pos
            elif '=' in move_string:
                at = re.findall('[a-h][1-8]', move_string)
                at = at[len(at) - 1]
                p = re.findall('=[QRBN]', move_string)[0]
                piece = self.get_piece_by_code(p[1])
                if desc is None:
                    desc = player + ' promotes pawn to ' + piece + ' at ' + at
                else:
                    desc += ' and promotes pawn to ' + piece + ' at ' + at
            else:
                at = re.findall('[a-h][1-8]', move_string)
                at = at[len(at) - 1]
                desc = player + ' moves ' + piece + ' to ' + at

            if '+' in move_string:
                desc = desc + ' and gives check'
            elif '#' in move_string:
                desc = desc + ' and checkmates'

            return desc


class Sprite():

    def __init__(self, img_location, width=None, height=None):
        self.img = img_location
        self.image = pyglet.resource.image(img_location)
        self.sprite = pyglet.sprite.Sprite(self.image)
        if width:
            self.width = width
        if height:
            self.height = height

    # properties
    @property
    def width(self):
        return self.sprite.width

    @property
    def height(self):
        return self.sprite.height

    @property
    def x(self):
        return self.sprite.x

    @property
    def y(self):
        return self.sprite.y

    @x.setter
    def x(self, value):
        self.sprite.x = value

    @y.setter
    def y(self, value):
        self.sprite.y = value

    @width.setter
    def width(self, value):
        self.sprite.scale_x = 1.0
        self.sprite.scale_x = value / self.sprite.width

    @height.setter
    def height(self, value):
        self.sprite.scale_y = 1.0
        self.sprite.scale_y = value / self.sprite.height

    def resize(self, size_x, size_y):
        self.width = size_x
        self.height = size_y

    def set_pos(self, pos):
        self.x = (pos % 8) * self.width
        self.y = (7 - (pos // 8)) * self.height

    def draw(self):
        self.sprite.draw()


class Board(Geom):

    def __init__(self, width=512, height=512, piece_arr=None, highlight_arr=None):
        Geom.__init__(self)
        self.bg = Sprite('Textures/Grey_Square.png', width + 32 + width, height)
        self.board = Sprite('Textures/Board.png', width, height)
        if piece_arr:
            self.pieces = piece_arr
        else:
            self.pieces = [None] * 64
        self.highlights = [None] * 64
        if highlight_arr:
            self.highlights = highlight_arr
        self.prom_area = Sprite('Textures/Dark_Square.png', width, height)
        self.prom_area.x = width + 32
        self.prom_piece = [None] * 4

    def update(self, move, highlight=False):
        self.clear_highlight()
        pos_from = move[0]
        pos_to = move[1]
        promotion = move[2]
        piece = self.pieces[pos_from]
        cap = self.pieces[pos_to]

        # Castling
        if 'King' in piece.img and abs(pos_from - pos_to) == 2:
            self.pieces[pos_from] = None
            self.set_at(pos=pos_to, piece=piece)
            if pos_to % 8 == 2:
                rook = self.pieces[pos_from - 4]
                self.set_at(pos=pos_from - 1, piece=rook)
            else:
                rook = self.pieces[pos_from + 3]
                self.set_at(pos=pos_from + 1, piece=rook)

        # en passant
        elif 'Pawn' in piece.img and pos_from % 8 != pos_to % 8 and cap is None:
            self.pieces[pos_from] = None
            self.set_at(pos=pos_to, piece=piece)
            if 'White' in piece.img:
                self.pieces[pos_to + 8] = None
            else:
                self.pieces[pos_to - 8] = None

        # Pawn Promotion
        elif promotion:
            self.pieces[pos_from] = None
            piece = self.get_piece(promotion)
            self.set_at(pos=pos_to, piece=piece)

        else:
            self.pieces[pos_from] = None
            self.set_at(pos=pos_to, piece=piece)

        if highlight:
            row_from = pos_from // 8
            row_to = pos_to // 8
            col_from = pos_from % 8
            col_to = pos_to % 8

            if row_from % 2 == col_from % 2:
                highlight_from = Sprite('Textures/Moved_Light.png', self.width/8, self.height/8)
            else:
                highlight_from = Sprite('Textures/Moved_Dark.png', self.width/8, self.height/8)

            if row_to % 2 == col_to % 2:
                highlight_to = Sprite('Textures/Moved_Light.png', self.width/8, self.height/8)
            else:
                highlight_to = Sprite('Textures/Moved_Dark.png', self.width/8, self.height/8)

            self.set_highlight(pos=pos_from, highlight=highlight_from)
            self.set_highlight(pos=pos_to, highlight=highlight_to)

    def get_piece(self, id):
        pieces = [None, 'Textures/White_King.png', 'Textures/White_Queen.png',
                  'Textures/White_Rook.png',   'Textures/White_Bishop.png',
                  'Textures/White_Knight.png', 'Textures/White_Pawn.png',
                  'Textures/Black_Pawn.png', 'Textures/Black_Knight.png',
                  'Textures/Black_Bishop.png', 'Textures/Black_Rook.png',
                  'Textures/Black_Queen.png', 'Textures/Black_King.png']
        return Sprite(pieces[id], self.width/8, self.height/8)

    def set_at(self, pos, piece=None):
        if not 0 <= pos < 64:
            return
        if piece is None:
            self.pieces[pos] = None
        else:
            self.pieces[pos] = piece
            piece.set_pos(pos)

    def set_highlight(self, pos, highlight=None):
        if not 0 <= pos < 64:
            return
        if highlight is None:
            self.highlights[pos] = None
        else:
            self.highlights[pos] = highlight
            highlight.set_pos(pos)

    def save_state(self):
        pieces = []
        highlights = []
        for i in range(64):
            pieces.append(self.pieces[i])
            highlights.append(self.highlights[i])
        return pieces, highlights

    def load_state(self, state):
        for i in range(64):
            self.set_at(pos=i, piece=state[0][i])
            self.set_highlight(pos=i, highlight=state[1][i])

    def render1(self):
        self.bg.draw()
        self.board.draw()
        self.prom_area.draw()
        for i in range(64):
            if self.highlights[i]:
                self.highlights[i].draw()
            if self.pieces[i]:
                self.pieces[i].draw()
        for j in range(4):
            if self.prom_piece[j]:
                self.prom_piece[j].draw()

    def reset(self):
        self.pieces.clear()
        self.highlights.clear()
        state = [-3, -5, -4, -2, -1, -4, -5, -3,
                 -6, -6, -6, -6, -6, -6, -6, -6,
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 '.', '.', '.', '.', '.', '.', '.', '.',
                 6, 6, 6, 6, 6, 6, 6, 6,
                 3, 5, 4, 2, 1, 4, 5, 3]
        for i in range(64):
            if state[i] == '.':
                self.pieces.append(None)
            else:
                piece = self.get_piece(state[i])
                self.pieces.append(piece)
                piece.set_pos(len(self.pieces) - 1)
            self.highlights.append(None)

    def clear_highlight(self):
        for i in range(64):
            self.highlights[i] = None

    def create_prom_menu(self, player):
        knight = self.get_piece(5 * player)
        bishop = self.get_piece(4 * player)
        rook = self.get_piece(3 * player)
        queen = self.get_piece(2 * player)
        self.prom_piece = [queen, rook, bishop, knight]
        for p in self.prom_piece:
            p.x = self.width + 32
        if player == 1:
            knight.y = self.height/2
            bishop.y = 5 * self.height/8
            rook.y = 6 * self.height/8
            queen.y = 7 * self.height/8
        else:
            knight.y = 3 * self.height/8
            bishop.y = 2 * self.height/8
            rook.y = self.height/8


    def clear_prom(self):
        for j in range(4):
            self.prom_piece[j] = None


    @property
    def width(self):
        return self.board.width

    @property
    def height(self):
        return self.board.height

    @width.setter
    def width(self, value):
        self.board.width = value

    @height.setter
    def height(self, value):
        self.board.height = value


class Player:

    def __init__(self, player=1, state=None):
        self.player = player
        self.move_handler = MoveHandler(p=player, state=state)
        self.fr = None
        self.to = None
        self.is_promotion = False
        self.moving_piece = None

    def play(self, action):
        print(action)
        return None

    def play1(self, pos, pr=None):
        if not 0 <= pos < 64:
            return None
        if self.fr is None and self.move_handler.state[pos] in self.move_handler.opponent:
            return None
        if self.fr is None and self.move_handler.state[pos] in ['.', 'x', 'o']:
            return None
        if self.fr is None:
            self.fr = pos
            self.moving_piece = self.move_handler.state[pos]
            return None
        elif self.to is None:
            self.to = pos
            if self.player == 1:
                if 0 <= pos < 8:
                    if self.moving_piece == 6:
                        self.is_promotion = True
                        return None
                    else:
                        self.is_promotion = False
                        self.move_handler.move(move=(self.fr, self.to, None))
                        fr = self.fr
                        to = self.to
                        self.fr = None
                        self.to = None
                        return fr, to, None
                else:
                    self.is_promotion = False
                    if (self.fr, self.to, None) not in self.move_handler.available_moves:
                        self.to = None
                        return None
                    self.move_handler.move(move=(self.fr, self.to, None), simulate=False)
                    fr = self.fr
                    to = self.to
                    self.fr = None
                    self.to = None
                    return fr, to, None
            else:
                if 56 <= pos < 64:
                    if self.moving_piece == -6:
                        self.is_promotion = True
                        return False
                    else:
                        self.is_promotion = False
                        self.move_handler.move(move=(self.fr, self.to, None))
                        fr = self.fr
                        to = self.to
                        self.fr = None
                        self.to = None
                        return fr, to, None
                else:
                    self.is_promotion = False
                    if (self.fr, self.to, None) not in self.move_handler.available_moves:
                        self.to = None
                        return None
                    self.move_handler.move(move=(self.fr, self.to, None), simulate=False)
                    fr = self.fr
                    to = self.to
                    self.fr = None
                    self.to = None
                    return fr, to, None
        elif self.is_promotion:
            if pr is None:
                return None
            else:
                self.is_promotion = False
                if (self.fr, self.to, abs(pr)*self.player) not in self.move_handler.available_moves:
                    return None
                self.move_handler.move(move=(self.fr, self.to, abs(pr)*self.player), simulate=False)
                fr = self.fr
                to = self.to
                self.fr = None
                self.to = None
                return fr, to, abs(pr)*self.player
        else:
            return None

    def move_to_action(self, move):
        for i in range(len(self.playing.move_handler.available_moves)):
            if self.playing.move_handler.available_moves[i] == move:
                return i
