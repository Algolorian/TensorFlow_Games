import os
import numpy as np

start = 0

ls = np.array(os.listdir('Recorded-control/'))
ls2 = np.array([])
for i in ls:
    ls2 = np.append(ls2, i[7:-4])
ls2 = ls2.astype('int')
ls2.sort()
# print(ls2)

for filenum in range(len(ls2) - 1):
    if ls2[filenum+1] != ls2[filenum] + 1:
        continue
    print(filenum, ls2[filenum], ls2[filenum + 1], start)
    os.rename(f"Recorded-control/record-{ls2[filenum + 1]}.csv", f"Compiled_Records/record-{start}.csv")
    os.rename(f"Recorded-DB/record-{ls2[filenum]}.npy", f"Compiled_Records/record-{start}.npy")
    start += 1
