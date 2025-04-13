import numpy as np

tmp = []
for i in range(1, 6):
    with open(f'record_{i}.txt') as fp:
        t_ = []
        for j in fp.readlines():
            t_.append(eval(j.split('/')[0])[2])
        tmp.append(t_)

tmp = np.array(tmp)
print(list(tmp.mean(axis=0)))
    