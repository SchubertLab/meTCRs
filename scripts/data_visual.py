import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

training_data = DataLoader.BatchSampler(16, 2, '../data/dl_iedb_train.csv', do_weight=True)

amounts = [len(el) for el in training_data.dataset.values()]
amounts.sort()
print(sum(amounts))
print(amounts)


cnt = Counter(amounts)
print(cnt)


validation_data = DataLoader.BatchSampler(16, 2, '../data/dl_iedb_val.csv')

amounts = [len(el) for el in validation_data.dataset.values()]
amounts.sort()
print(sum(amounts))
print(amounts)


cnt = Counter(amounts)
print(cnt)