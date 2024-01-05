import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

df = pd.read_csv('data/ibd_lineages.csv')
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

df = df.head(6)
df = pd.DataFrame(df)

fig, ax = plt.subplots()

ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(4, 7)

for cell in [table[0, 0], table[0, 1],table[0,2],table[0,3],table[0,4],table[0,5],table[0,6],table[0,7]]:
    cell.get_text().set_weight('bold')
    cell.get_text().set_fontsize(14)
    cell.set_facecolor('#ADD8E6')

plt.savefig('dataframe_table.png', bbox_inches='tight', pad_inches=0.2, format='png')

plt.show()


toy_richness = np.sum(counts != 0, axis=0)
print(toy_richness)


toy_richness = [5, 8, 3, 6, 7, 9]

groups = ['A'] * 5 + ['B'] * 5

statistic, p_value = stats.wilcoxon(toy_richness, correction=True)

print(f"Wilcoxon Statistic: {statistic}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The difference is statistically significant (p < 0.05).")
else:
    print("The difference is not statistically significant (p >= 0.05).")


num_samples = 5
num_species = 10

counts = np.random.randint(0, 10, size=(num_species, num_samples))

singletons = np.sum(counts == 1, axis=0)
doubletons = np.sum(counts == 2, axis=0)

rares = singletons / (2 * doubletons)
rares[doubletons == 0] = 0

toy_richness = np.sum(counts > 0, axis=0)
chao1 = toy_richness + rares

print("Chao 1 Estimator:")
print(chao1)

sums = np.sum(counts, axis=0)

norm_counts = counts.copy()

for i in range(counts.shape[1]):
    norm_counts[:, i] = counts[:, i] / sums[i]

def shannon_div(vector):
    vector = vector * np.log(vector)
    vectorsum = np.nansum(vector) 
    return -vectorsum

shannon_diversities = np.apply_along_axis(shannon_div, axis=0, arr=norm_counts)

print("Shannon Diversities:")
print(shannon_diversities)
sums = np.sum(counts, axis=0)

norm_counts = counts.copy()

for i in range(counts.shape[1]):
    norm_counts[:, i] = counts[:, i] / sums[i]

simpson_indices = np.sum(np.square(norm_counts), axis=0)

inverse_simpson_indices = 1 / simpson_indices

print("Simpson Indices:")
print(simpson_indices)

count_vector = df.apply(pd.Series.nunique)

# Call the rarefaction function
def rarefaction(count_vector, depth):
    probab = count_vector / sum(count_vector)
    return np.random.multinomial(depth, probab, size=1)


result = rarefaction(count_vector, 100)
print(result)



