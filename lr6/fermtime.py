import time
from multiprocessing import Pool, cpu_count
from threading import Thread
from queue import Queue

from ferma_fact import fermat_factorization as cython_fermat


def python_fermat(N):
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

TEST_LST = [
    101, 9973, 104729, 101909,
    609133, 1300039, 9999991,
    99999959, 99999971,
    3000009, 700000133
] * 3  

def run_multiprocessing(func, data, workers=4):
    with Pool(workers) as pool:
        res = pool.map(func, data)
    return res



def run_multithreading(func, data, workers=4):
    q_in = Queue()
    q_out = Queue()

    for item in data:
        q_in.put(item)

    def worker():
        while not q_in.empty():
            try:
                item = q_in.get_nowait()
                result = func(item)
                q_out.put(result)
            except:
                break

    threads = [Thread(target=worker) for _ in range(workers)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    return list(q_out.queue)



def timed_run(name, func, *args):
    start = time.perf_counter()
    func(*args)
    end = time.perf_counter()
    print(f"{name} time: {end - start:.4f} sec")
    return end - start



if __name__ == "__main__":
    from matplotlib import pyplot as plt

    workers = cpu_count()

    times = {
        "Python + Threads": timed_run("Python + Threads", run_multithreading, python_fermat, TEST_LST, workers),
        "Python + Processes": timed_run("Python + Processes", run_multiprocessing, python_fermat, TEST_LST, workers),
        "Cython + Threads": timed_run("Cython + Threads", run_multithreading, cython_fermat, TEST_LST, workers),
        "Cython + Processes": timed_run("Cython + Processes", run_multiprocessing, cython_fermat, TEST_LST, workers),
    }

    plt.figure(figsize=(10, 6))
    plt.bar(times.keys(), times.values(), color=['red', 'orange', 'green', 'blue'])
    plt.ylabel("Время выполнения (сек)")
    plt.title("Сравнение: Потоки vs Процессы для Python и Cython реализаций")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("parallel_comparison.png")
    plt.show()
