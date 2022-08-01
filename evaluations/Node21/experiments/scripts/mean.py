import numpy as np

x = [0.46053, 0.33226, 0.49411, 0.43576, 0.49046]
y = [0.27606, 0.39016, 0.31149, 0.24076, 0.21543]

print("X - median: {}, min: {}, max: {}".format(np.median(x), np.min(x), np.max(x)))
print("Y - median: {}, min: {}, max: {}".format(np.median(y), np.min(y), np.max(y)))
print(np.median(x) - np.median(y))
