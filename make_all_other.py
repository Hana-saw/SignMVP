import numpy as np

data = np.load('sequence_dataset.npz')
X = data['X']
y = np.array(['other'] * len(data['y']))

np.savez('sequence_dataset_all_other.npz', X=X, y=y)
print('すべてのラベルをotherにしたデータセットを sequence_dataset_all_other.npz に保存しました')