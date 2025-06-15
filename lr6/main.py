import timeit
import matplotlib.pyplot as plt

from ferma_fact import fermat_factorization

TEST_LST = [
    101, 9973, 104729, 101909,
    609133, 1300039, 9999991,
    99999959, 99999971,
    3000009, 700000133
]

def python_version(N):
    import math
    def is_perfect_square(n):
        root = int(math.isqrt(n))
        return root * root == n

    if N % 2 == 0:
        return 2, N // 2

    x = math.isqrt(N) + 1
    while True:
        y_squared = x * x - N
        if is_perfect_square(y_squared):
            y = int(math.isqrt(y_squared))
            return x - y, x + y
        x += 1


# Python
py_time = timeit.timeit(
    "res = [python_version(i) for i in TEST_LST]",
    setup="from __main__ import python_version, TEST_LST",
    number=10
)

# Cython
cy_time = timeit.timeit(
    "res = [fermat_factorization(i) for i in TEST_LST]",
    setup="from __main__ import fermat_factorization, TEST_LST",
    number=10
)

print(f"Python time: {py_time:.4f} sec")
print(f"Cython time: {cy_time:.4f} sec")

# График
plt.bar(["Python", "Cython"], [py_time, cy_time], color=["red", "green"])
plt.ylabel("Время выполнения (сек)")
plt.title("Сравнение скорости: Python vs Cython")
plt.savefig("comparison.png")
plt.show()
