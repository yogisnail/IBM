import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.stats import spearmanr
from scipy.stats import ranksums
from scipy.stats import wilcoxon

ibd_metadata = pd.read_csv('data/ibd_metadata.csv', nrows=150)
ibd_taxa = pd.read_csv('data/ibd_taxa.csv', nrows=150)

ibd_taxa = ibd_taxa.fillna(1e-10)
ibd_taxa[ibd_taxa == 0] = 1e-10

def shannon_div(vec):
    vec = pd.to_numeric(vec, errors='coerce')
    p = vec / np.nansum(vec)
    return -np.nansum(p * np.log(p))

ibd_metadata['shannon'] = np.apply_along_axis(shannon_div, 1, ibd_taxa)

def pielou_evenness(vec):
    vec = pd.to_numeric(vec, errors='coerce')
    shannon_val = shannon_div(vec)
    species_count = np.sum(~np.isnan(vec))
    shannon_max = np.log(species_count)
    return shannon_val / shannon_max

ibd_metadata['pielou'] = np.apply_along_axis(pielou_evenness, 1, ibd_taxa)

def simpson_div(vec):
    vec = pd.to_numeric(vec, errors='coerce')
    return 1 - np.nansum((vec / np.nansum(vec)) ** 2)

ibd_metadata['simpson'] = np.apply_along_axis(simpson_div, 1, ibd_taxa)

def invsimpson_index(vec):
    vec = pd.to_numeric(vec, errors='coerce')
    return 1 / np.nansum((vec / np.nansum(vec)) ** 2)

ibd_metadata['invsimpson'] = np.apply_along_axis(invsimpson_index, 1, ibd_taxa)

plt.figure(figsize=(8, 6))
plt.boxplot([ibd_metadata[ibd_metadata['Diagnosis'] == 'CD']['shannon'],
             ibd_metadata[ibd_metadata['Diagnosis'] == 'UC']['shannon'],
             ibd_metadata[ibd_metadata['Diagnosis'] == 'Control']['shannon']],
            labels=['CD', 'UC', 'Control'])
plt.xlabel('Diagnosis')
plt.ylabel('Shannon Diversity')
plt.title('Boxplot of Shannon Diversity Index by Diagnosis')
plt.savefig('shannon_diversity_boxplot.png')
plt.show()
plt.close()

plt.scatter(ibd_metadata['Age'], ibd_metadata['shannon'])
plt.xlabel('Age')
plt.ylabel('Shannon Diversity')
plt.title('Scatter Plot of Shannon Diversity vs. Age')
plt.show()
plt.savefig('Scatter Plot of Shannon Diversity vs. Age.png')
plt.close()

spearman_corr, _ = spearmanr(ibd_metadata['Age'], ibd_metadata['shannon'])
print(f"Spearman Correlation: {spearman_corr}")


spearman_corr, p_value = spearmanr(ibd_metadata['Age'], ibd_metadata['shannon'])
print(f"Spearman Correlation: {spearman_corr}")
print(f"P-value: {p_value}")


fp_index = np.where(ibd_taxa.iloc[:, 0] == "Faecalibacterium_prausnitzii")[0][0]

control_mask = (ibd_metadata['Diagnosis'] == "Control").astype(int)
disease_mask = ((ibd_metadata['Diagnosis'] == "CD") | (ibd_metadata['Diagnosis'] == "UC")).astype(int)


control_data = ibd_taxa.loc[control_mask, ibd_taxa.columns[fp_index]]
disease_data = ibd_taxa.loc[disease_mask, ibd_taxa.columns[fp_index]]

statistic, p_value = wilcoxon(control_data, disease_data, zero_method='zsplit')

print("Wilcoxon Rank-Sum Test:")
print(f"Statistic: {statistic}")
print(f"P-value: {p_value}")


col2 = "#FF0000"
numeric_data = pd.to_numeric(ibd_taxa.iloc[fp_index, :], errors='coerce').dropna()

sns.histplot(
    data=numeric_data,
    bins='fd',
    color=col2,
    stat="density",
    common_norm=False,
    kde=False,
    element="step",
    linewidth=2,
    label="Control"
)

plt.legend(title="Diagnosis", loc="upper right", labels=["Control", "IBD"])
plt.xlabel("Abundance")
plt.ylabel("Density")
plt.show()
plt.savefig('Diagnosis.png')
plt.close()



