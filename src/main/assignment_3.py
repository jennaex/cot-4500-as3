import numpy as np


# setting up the function for question 1 & 2
def func(t, y):
    return t - y ** 2


# Function for euler formula
def eulermethod(t0, y, h, t):
    tempvariable = -0

    while t0 < t:
        tempvariable = y
        y = y + h * func(t0, y)
        t0 = t0 + h

    return y


# Function for Runge Kutta Method
def rungekuttamethod(t0, y0, t, h):
    n = int((t - t0) / h)
    y = y0
    for i in range(1, n + 1):
        m1 = h * func(t0, y)
        m2 = h * func(t0 + 0.5 * h, y + 0.5 * m1)
        m3 = h * func(t0 + 0.5 * h, y + 0.5 * m2)
        m4 = h * func(t0 + h, y + m3)

        y = y + (1.0 / 6.0) * (m1 + 2 * m2 + 2 * m3 + m4)

        t0 = t0 + h
    return y


# Function for gauss elimation and backwards subsitution
def gaussandbackwards(a, b):
    n = len(b)
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if a[i, k] != 0.0:
                iii = a[i, k] / a[k, k]
                a[i, k + 1:n] = a[i, k + 1:n] - iii * a[k, k + 1:n]
                b[i] = b[i] - iii * b[k]
                # the backward substitution part
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(a[k, k + 1:n], b[k + 1:n])) / a[k, k]

    return b

# Function for LU Factorization
def lufactorization(array):
    r = array.shape[0]
    U = array.copy()
    L = np.eye(r, dtype=np.double)
    for i in range(r):
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]
    return L, U

# Function to see if the matrix is diagnally dominate
def diagnallydom(q, k):
    for i in range(0, k):
        rowsum = 0
        for j in range(0, k):
            rowsum = rowsum + abs(q[i][j])

        rowsum = rowsum - abs(q[i][i])

        # Checking the conditions:
        if abs(q[i][i]) < rowsum:
            return False

    return True


if __name__ == '__main__':
    # Question 1:
    # Values needed for the Euler's Equation
    t0 = 0
    y0 = 1
    h = 0.20
    t = 1.80

    print("%.5f" % eulermethod(t0, y0, h, t), "\n")

    # Question 2
    # Values needed for the Runge Kutta Method
    x0 = 0
    y = 1
    x = 2
    h = 0.2

    print("%.5f" % rungekuttamethod(x0, y, x, h), "\n")

    # Question 3
    # creating the array
    a = np.array([[2.0, -1.0, 1.0], [1.0, 3.0, 1.0], [-1.0, 5.0, 4.0]])
    b = np.array([6.0, 0.0, -3.0])
    x = gaussandbackwards(a, b)

    print(x, "\n")

    # Question 4a
    # creating the array
    n_array = np.array([[1.0, 1.0, 0.0, 3.0], [2.0, 1.0, -1.0, 1.0], [3.0, -1.0, -1.0, 2.0], [-1.0, 2.0, 3.0, -1.0]])
    # calculating the determinant of matrix
    det = np.linalg.det(n_array)

    print("%.5f" % round(det), "\n")

    # Question 4b & 4c
    L, U = lufactorization(n_array)

    print(L, "\n")
    print(U, "\n")

    # Question 5
    k = 5
    l = [[9, 0, 5, 2, 1],
         [3, 9, 1, 2, 1],
         [0, 1, 7, 2, 3],
         [4, 2, 3, 12, 2],
         [3, 2, 4, 0, 8]]

    if diagnallydom(l, k):
        print("True", "\n")
    else:
        print("False", "\n")

    # Question 6
    # Creating the array
    matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])

    # A matrix is positive definite if the eigenvalues are positive
    # Checking that the eigenvalues are positive
    eigen = np.all(np.linalg.eigvals(matrix) > 0)

    # Condition statement to show if the matrix is positive definite
    if eigen:
        print(eigen, "\n")
    else:
        print(eigen, "\n")
