import numpy as np
import bcolz
bz = bcolz.carray(np.array([], dtype="S500"))
for i in range(1000):
    bz.append(np.array(["xxx"]))
    if len(bz) >= 2 and type(bz[1]) == np.ndarray:
        print("oops")
