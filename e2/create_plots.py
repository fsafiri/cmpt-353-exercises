import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('pagecounts-20190509-120000.txt', sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])

data=data.sort_values(by='views',ascending=False )

a=data['views'].count()
x_axis=np.arange(0,a,1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1) 
plt.plot(x_axis,data['views'].values) 
plt.xlabel("Rank")
plt.ylabel("Views")
#plt.show()

data_2=pd.read_csv('pagecounts-20190509-130000.txt', sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])
#data_2

a=data_2['views']
a=a.to_frame()

df3 = pd.merge(data, a, left_index=True, right_index=True)

plt.subplot(1, 2, 2)
f=plt.scatter(df3['views_x'],df3['views_y'] )
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Hour 1 views")
plt.ylabel("Hour 2 views")

plt.savefig('wikipedia.png')
print('pic is generated & saved... ')


