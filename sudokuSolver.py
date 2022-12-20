# For this program, I assume a solvable sudoku puzzle.

# Test Board
board = [
    [6,4,0,0,0,7,0,0,0],
    [0,0,0,4,9,0,0,2,0],
    [0,9,1,0,0,5,0,0,0],
    [8,0,0,0,0,6,1,0,0],
    [0,5,0,0,0,0,0,6,0],
    [0,0,2,5,0,0,0,0,7],
    [0,0,0,3,0,0,5,4,0],
    [0,8,0,0,4,2,0,0,0],
    [0,0,0,7,0,0,0,9,3]
]

# solve(board) takes in a board and solves the board using a recursive backtracking algorithm

def solve(board):
    empty_square = find_empty(board)

    if not empty_square:
        return True
    else:
        row, column = empty_square

        for i in range(1, 10):
            if check_valid(board, i, row, column):
                board[row][column] = i

                if solve(board):
                    return True

            board[row][column] = 0
    
    return False

# find_empty(board) takes in a board and returns the (row, column) tuple of an empty square, if one exists, and returns None otherwise.

def find_empty(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)
    return None

# check_valid(board, num, row, column) takes in a board, a number, and a row/column position, and determines if the number can be placed 
# within the given row/column position.

def check_valid(board, num, row, column):
    # Checks row
    for i in range(len(board[row])):
        if num == board[row][i] and i != column:
            return False
    
    # Checks column
    for i in range(len(board)):
        if num == board[i][column] and i != row:
            return False
    
    # Checks Box
    box_x = column // 3
    box_y = row // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != (row, column):
                return False

    return True

#solve(board)
#print(board)