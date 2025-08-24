import numpy as np
from collections import Counter

data = np.load('sequence_dataset.npz')
y = data['y']
print(Counter(y))