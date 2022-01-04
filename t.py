# %%
import torch
from torch.multiprocessing import Pool

ctx = torch.multiprocessing.get_context("spawn")
f_list = [i for i in range(100)]

t = 0
class A():
    def __init__(self):
        print('实例化了类A\n')
a = A()

def fn(x):
    a.a = x
    return t

def test():
    global t
    t = 1
    with ctx.Pool(10) as pool:
        for _ in pool.imap_unordered(fn, f_list):
            pass
            # print(_)

# %%
if __name__ == '__main__':
    test()

# %%
