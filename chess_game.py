import pygame
import sys

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
BLUE = (0, 0, 255)

# Chess Unicode symbols
PIECES = {
    'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚', 'p': '♟',
    'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔', 'P': '♙'
}

# Font setup
FONT = pygame.font.SysFont('segoeuisymbol', 64)

class ChessBoard:
    def __init__(self):
        self.board = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"]
        ]
        self.turn = "white"
        self.selected_piece = None
        self.valid_moves = []

    def draw_board(self, win):
        for row in range(ROWS):
            for col in range(COLS):
                color = WHITE if (row + col) % 2 == 0 else GRAY
                pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                piece = self.board[row][col]
                if piece != " ":
                    text = FONT.render(PIECES[piece], True, BLACK if piece.isupper() else BLACK)
                    win.blit(text, (col * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_width() // 2,
                                    row * SQUARE_SIZE + SQUARE_SIZE // 2 - text.get_height() // 2))

        # Highlight selected piece and valid moves
        if self.selected_piece:
            row, col = self.selected_piece
            pygame.draw.rect(win, BLUE, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)
            for move in self.valid_moves:
                move_row, move_col = move
                pygame.draw.circle(win, BLUE, (move_col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                               move_row * SQUARE_SIZE + SQUARE_SIZE // 2), 10)

    def select_piece(self, row, col):
        piece = self.board[row][col]
        if piece != " " and ((self.turn == "white" and piece.isupper()) or (self.turn == "black" and piece.islower())):
            self.selected_piece = (row, col)
            self.valid_moves = self.get_valid_moves(row, col)
        else:
            self.selected_piece = None
            self.valid_moves = []

    def move_piece(self, row, col):
        if self.selected_piece and (row, col) in self.valid_moves:
            start_row, start_col = self.selected_piece
            piece = self.board[start_row][start_col]
            self.board[start_row][start_col] = " "
            self.board[row][col] = piece
            self.turn = "black" if self.turn == "white" else "white"
            self.selected_piece = None
            self.valid_moves = []

    def get_valid_moves(self, start_row, start_col):
        piece = self.board[start_row][start_col]
        moves = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.is_valid_move(piece, start_row, start_col, row, col):
                    moves.append((row, col))
        return moves

    def is_valid_move(self, piece, start_row, start_col, end_row, end_col):
        # Check if the destination square is occupied by a piece of the same color
        destination_piece = self.board[end_row][end_col]
        if destination_piece != ' ' and destination_piece.isupper() == piece.isupper():
            return False

        if piece.lower() == 'p':
            return self.is_valid_pawn_move(piece, start_row, start_col, end_row, end_col)
        elif piece.lower() == 'r':
            return self.is_valid_rook_move(start_row, start_col, end_row, end_col)
        elif piece.lower() == 'n':
            return self.is_valid_knight_move(start_row, start_col, end_row, end_col)
        elif piece.lower() == 'b':
            return self.is_valid_bishop_move(start_row, start_col, end_row, end_col)
        elif piece.lower() == 'q':
            return self.is_valid_queen_move(start_row, start_col, end_row, end_col)
        elif piece.lower() == 'k':
            return self.is_valid_king_move(start_row, start_col, end_row, end_col)
        return False

    def is_valid_pawn_move(self, piece, start_row, start_col, end_row, end_col):
        direction = -1 if piece.isupper() else 1
        start_row_white = 6 if piece.isupper() else 1
        if start_col == end_col:
            if self.board[end_row][end_col] == ' ':
                if start_row + direction == end_row:
                    return True
                if start_row == start_row_white and start_row + 2 * direction == end_row and self.board[start_row + direction][start_col] == ' ':
                    return True
        elif abs(start_col - end_col) == 1 and start_row + direction == end_row:
            if self.board[end_row][end_col] != ' ' and self.board[end_row][end_col].isupper() != piece.isupper():
                return True
        return False

    def is_valid_rook_move(self, start_row, start_col, end_row, end_col):
        if start_row != end_row and start_col != end_col:
            return False
        if start_row == end_row:
            step = 1 if start_col < end_col else -1
            for col in range(start_col + step, end_col, step):
                if self.board[start_row][col] != ' ':
                    return False
        else:
            step = 1 if start_row < end_row else -1
            for row in range(start_row + step, end_row, step):
                if self.board[row][start_col] != ' ':
                    return False
        return True

    def is_valid_knight_move(self, start_row, start_col, end_row, end_col):
        return (abs(start_row - end_row), abs(start_col - end_col)) in [(2, 1), (1, 2)]

    def is_valid_bishop_move(self, start_row, start_col, end_row, end_col):
        if abs(start_row - end_row) != abs(start_col - end_col):
            return False
        row_step = 1 if start_row < end_row else -1
        col_step = 1 if start_col < end_col else -1
        for i in range(1, abs(start_row - end_row)):
            if self.board[start_row + i * row_step][start_col + i * col_step] != ' ':
                return False
        return True

    def is_valid_queen_move(self, start_row, start_col, end_row, end_col):
        return self.is_valid_rook_move(start_row, start_col, end_row, end_col) or self.is_valid_bishop_move(start_row, start_col, end_row, end_col)

    def is_valid_king_move(self, start_row, start_col, end_row, end_col):
        return max(abs(start_row - end_row), abs(start_col - end_col)) == 1

def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess')
    chess_board = ChessBoard()
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // SQUARE_SIZE, pos[0] // SQUARE_SIZE
                if chess_board.selected_piece:
                    chess_board.move_piece(row, col)
                else:
                    chess_board.select_piece(row, col)

        chess_board.draw_board(win)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()