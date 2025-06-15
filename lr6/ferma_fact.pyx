# ferma_fact.pyx
from libc.math cimport sqrt

cdef inline bint is_perfect_square(long long n):
    cdef long long root = <long long>sqrt(n)
    return root * root == n

def fermat_factorization(long long N):
    cdef long long x, y, y_squared
    if N % 2 == 0:
        return 2, N // 2

    x = <long long>sqrt(N) + 1
    while True:
        y_squared = x * x - N
        if is_perfect_square(y_squared):
            y = <long long>sqrt(y_squared)
            return x - y, x + y
        x += 1
