from __future__ import annotations
from random import random
from pprint import pprint


class Matrix:
    def __init__(self, rows: int, columns: int):
        self.rows = rows
        self.columns = columns

        self.data = [[0]*columns for i in range(rows)]

    def COPY(self, matrix: Matrix):
        # "from" is a Python keyword
        if self.columns != matrix.columns or self.rows != matrix.rows:
            raise Exception("Matrices must have the same dimension to copy")

        if self is matrix:
            raise Exception("Cannot copy a matrix to itself, that's retarded")

        self.data = matrix.data.copy()

    def Transpose(self, matrix: Matrix):
        self.rows = matrix.columns
        self.columns = matrix.rows
        new_data = [[0]*matrix.rows for i in range(matrix.columns)]
        for r in range(matrix.rows):
            for c in range(matrix.columns):
                new_data[c][r] = matrix.data[r][c]
        self.data = new_data

    def mat_MUL(self, matrix1: Matrix, matrix2: Matrix):
        if matrix1.columns != matrix2.rows:
            raise Exception("Number of columns of matrix 1 must be equal to rows of matrix 2")

        if self is matrix1 or self is matrix2:
            raise Exception("Destination matrix must be distinct from matrix 1 and matrix 2")

        sharedDim: int = matrix1.columns

        for r in range(self.rows):
            for c in range(matrix2.columns):
                total: float = 0
                for d in range(sharedDim):
                    total += matrix1.data[r][d] * matrix2.data[d][c]
                self.data[r][c] = total

    def scalar_MUL(self, scalar: float):
        for r in range(self.rows):
            for c in range(self.columns):
                self.data[r][c] *= scalar

    def Hadamard(self, matrix: Matrix):
        if self.rows != matrix.rows or self.columns != matrix.columns:
            raise Exception("Matrices must be of same dimension in order to calculate Hadamard product")

        for r in range(self.rows):
            for c in range(self.columns):
                self.data[r][c] *= matrix.data[r][c]

    def ADD(self, matrix: Matrix):
        if self.rows != matrix.rows or self.columns != matrix.columns:
            raise Exception("Matrices must be of same dimension in order to perform addition")

        for r in range(self.rows):
            for c in range(self.columns):
                self.data[r][c] += matrix.data[r][c]

    def SUB(self, matrix: Matrix):
        if self.rows != matrix.rows or self.columns != matrix.columns:
            raise Exception("Matrices must be of same dimension in order to perform subtraction")

        for r in range(self.rows):
            for c in range(self.columns):
                self.data[r][c] -= matrix.data[r][c]

    def GetAverageColumnVector(self, matrix: Matrix):
        for r in range(self.rows):
            # "sum" is a built-in function in python
            total: float = 0
            for c in range(self.columns):
                total += self.data[r][c]

            matrix.data[r][0] = total / self.columns

    def Randomize(self):
        for r in range(self.rows):
            for c in range(self.columns):
                # random() returns a float in the range [0, 1).
                # Not having 1 as a possible output should not be a issue
                self.data[r][c] = random()

    def PrintMatrix(self):
        pprint(self.data)

    def AddColumnVector(self, columnVector: Matrix):
        if self.rows != columnVector.rows or columnVector.columns != 1:
            raise Exception("Column vector must have the same number of rows as the matrix and a single column")

        for c in range(self.columns):
            for r in range(self.rows):
                self.data[r][c] += columnVector.data[r][0]

    def SubColumnVector(self, columnVector: Matrix):
        if self.rows != columnVector.rows or columnVector.columns != 1:
            raise Exception("Column vector must have the same number of rows as the matrix and a single column")

        for c in range(self.columns):
            for r in range(self.rows):
                self.data[r][c] -= columnVector.data[r][0]

    def ApplyFunction(self, matrix: Matrix, function):
        for r in range(self.rows):
            for c in range(self.columns):
                matrix.data[r][c] = function(self.data[r][c])
