import numpy as np


class SudokuSolver:
    """
    """
    


    def find_empty(self, sudu) -> tuple[int, int]:
        """
        """

        for i in range(len(sudu)):
            for j in range(len(sudu[0])):
                if sudu[i][j] == 0:
                    return (i, j)  # row, col

        return None


    def cross_check(self, sudu:np.ndarray, num, pos)->bool:
        # Check row
        for i in range(len(sudu[0])):
            if sudu[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(sudu)):
            if sudu[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if sudu[i][j] == num and (i,j) != pos:
                    return False

        return True


    def solver(self, sudu: np.ndarray):
        find= self.find_empty(sudu)
        if not  find:
            return True
        else:
            row, col = find

        for i in range(1,10):
            if self.cross_check(sudu, i, (row, col)):
                sudu[row][col] = i

                if self.solver(sudu):
                    return True

                sudu[row][col] = 0

        return False

    def print_board(self,sudu:np.ndarray):

        for i in range(len(sudu)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - - - ")

            for j in range(len(sudu[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")

                if j == 8:
                    print(sudu[i][j])
                else:
                    print(str(sudu[i][j]) + " ", end="")



    