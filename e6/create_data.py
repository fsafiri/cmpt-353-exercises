import time
from implementations import all_implementations
import pandas as pd
import numpy as np
def main():
    num_tests = 50
    data = pd.DataFrame()
    for sort in all_implementations:
        algorithm = sort.__name__
        run_times = np.empty(num_tests)
        for i in range(num_tests):   
            random_array = np.random.randint(900000, size=1000) 
            st = time.time()
            res=sort(random_array)
            en = time.time()
            run_times[i] = en - st

        data[algorithm] = run_times

    data.to_csv('data.csv', index=False)

if __name__ == '__main__':
    main()
    #print('done')