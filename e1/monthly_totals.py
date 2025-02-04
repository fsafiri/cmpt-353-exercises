import numpy as np
import pandas as pd

def get_precip_data():
    return pd.read_csv('precipitation.csv', parse_dates=[2])


def date_to_month(d):
    
    return '%04i-%02i' % (d.year, d.month)


def pivot_months_pandas(data):
    
    grouped_data=data.groupby(['name','month'])
    t=grouped_data.aggregate({'precipitation': 'sum'}).reset_index()
    c=grouped_data.aggregate({'precipitation': 'count'}).reset_index()
    
    totals=t.pivot(index='name', columns='month', values='precipitation')
    counts=c.pivot(index='name', columns='month',values='precipitation')
    
    return totals, counts

#test=pivot_months_pandas(data)
#test

def main():
    data = get_precip_data()
    
    data["month"]=data["date"].apply(date_to_month)
    
    totals, counts = pivot_months_pandas(data)
    totals.to_csv('totals.csv')
    counts.to_csv('counts.csv')
    np.savez('monthdata.npz', totals=totals.values, counts=counts.values)


if __name__ == '__main__':
    main()