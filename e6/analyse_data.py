import pandas as pd
from scipy import stats
import itertools #https://docs.python.org/3/library/itertools.html
import sys

data = pd.read_csv(sys.argv[1])

    
# anova test 
anova_result = stats.f_oneway(*[data[col] for col in data.columns])
print("ANOVA result:", anova_result)

# ttests with bonferroni correction
num_comparisons = len(list(itertools.combinations(data.columns, 2)))
alpha = 0.05
bonferroni_alpha = alpha / num_comparisons
print(f"\n bonferroni corrected alpha for {num_comparisons} comparison: {bonferroni_alpha}")

#store results of pairwise ttest
pairwise_results = {}

for (alg1, alg2) in itertools.combinations(data.columns, 2):
    t_stat, p_value = stats.ttest_ind(data[alg1], data[alg2])
    pairwise_results[(alg1, alg2)] = (t_stat, p_value)
    print(f"\nttest between {alg1} & {alg2}: p-value = {p_value}")

# ranking of algos on mean runtime
means = data.mean().sort_values()
print("\nranking of algorithms based on mean run time:")
print(means)
