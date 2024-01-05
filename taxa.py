import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


df = pd.read_csv("data/ibd_taxa.csv", index_col=0)

print(type(df))
print(df.head)
print(df.describe().round())

size_A = 100
prob_A = [0.2, 0.4, 0.05, 0.02, 0.15, 0.13, 0.01, 0.04]

A = np.random.multinomial(size_A, prob_A, 5)

size_B = 100
prob_B = [0.6, 0.25, 0, 0.04, 0.02, 0.06, 0.02, 0]

B = np.random.multinomial(size_B, prob_B, 5)

counts = np.column_stack((A, B))

groups = np.array(['A'] * 5 + ['B'] * 5)

print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Combined Matrix (counts):")
print(counts)
print("Sample Groups (groups):")
print(groups)

num_samples = 10
num_phyla = 3

ibd_taxa = np.random.randint(50, 200, size=(num_phyla, num_samples))

colors = np.random.rand(num_phyla, 3)

fig, ax = plt.subplots()

for i, row in enumerate(ibd_taxa):
    ax.bar(np.arange(len(row)) + i * 0.2, row, width=0.2, label=f'Phylum {i + 1}', color=colors[i])

ax.set_xlabel('Samples')
ax.set_ylabel('Counts')
ax.legend()
plt.savefig('barplot.png')
plt.show()
plt.close()


def jaccard_sets(a, b):
    taxa = list(map(str, range(1, len(a) + 1)))
    aset = set(taxa[i] for i, val in enumerate(a) if val != 0)
    bset = set(taxa[i] for i, val in enumerate(b) if val != 0)
    return 1 - len(aset.intersection(bset)) / len(aset.union(bset))

example_a = np.array([1, 0, 1, 0, 0])
example_b = np.array([0, 1, 1, 0, 1])

jaccard_distance = jaccard_sets(example_a, example_b)
print(f"Jaccard Distance: {jaccard_distance}")


#Principal Coordinates Analysis




