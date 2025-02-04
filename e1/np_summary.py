import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']


min_tot=np.argmin(np.sum(totals,axis=1))
min_tot

tot_precip=np.sum(totals, axis=0)

tot_observ=np.sum(counts, axis=0)

avg_arr = np.divide(tot_precip, tot_observ)
print("Average precipitation in each month:", avg_arr)

tot_precip=np.sum(totals, axis=1)

tot_observ=np.sum(counts, axis=1)

avg_arr = np.divide(tot_precip, tot_observ)

print("Average precipitation in each city:", avg_arr)

#You can assume the number of columns will be divisible by 3. Hint
num_months = totals.shape[1]
num_quarters = num_months//3

reshaped_tot = totals.reshape(totals.shape[0], num_quarters, 3)
quarter_tot = reshaped_tot.sum(axis=2)

print("Quarterly precipitation totals\n:", quarter_tot)