import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name'])
counts=  pd.read_csv('counts.csv').set_index(keys=['name'])

sum=totals.sum(axis=1)
min_index = sum.idxmin()
print("City with lowest total precipitation:", min_index)

tot_precip=totals.sum(axis=0)
tot_observ=counts.sum(axis=0)
avg_prep=tot_precip.div(tot_observ)
print("Average precipitation in each month:\n",avg_prep)

tot_precip=totals.sum(axis=1)
tot_observ=counts.sum(axis=1)
avg_prep=tot_precip.div(tot_observ)
print("Average precipitation in each city::\n",avg_prep)
