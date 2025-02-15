import pandas as pd 
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy import stats
import sys
import datetime as dt



OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)



def main():
    

    counts = pd.read_json(sys.argv[1], lines=True)
#counts 

    counts=counts[counts['date'].dt.year.isin([2012, 2013])]
    counts= counts[counts['subreddit'] == 'canada']

    def day_filter(counts):
        weekends = counts[counts['date'].dt.weekday.isin([5, 6])]
        weekdays = counts[~counts['date'].dt.weekday.isin([5, 6])]    
        return weekends, weekdays

    
    weekends, weekdays = day_filter(counts)
    
    #tudent ttest
    initial_ttest_p = stats.ttest_ind(weekdays['comment_count'], weekends['comment_count']).pvalue
    initial_weekend_normality_p=stats.normaltest(weekends['comment_count']).pvalue
    initial_weekday_normality_p=stats.normaltest(weekdays['comment_count']).pvalue
    initial_levene_p= stats.levene(weekends['comment_count'],weekdays['comment_count']).pvalue
    #sqrt
    transformed_weekday=np.sqrt(weekdays['comment_count'])
    transformed_weekend=np.sqrt(weekends['comment_count'])
    transformed_weekend_normality_p=stats.normaltest(weekends['comment_count']).pvalue
    transformed_weekday_normality_p =stats.normaltest(weekdays['comment_count']).pvalue
    transformed_levene_p = stats.levene(transformed_weekend,transformed_weekday).pvalue
    #iso week yeear
    
    weekends_iso = weekends['date'].dt.isocalendar()[['year', 'week']]
    weekdays_iso = weekdays['date'].dt.isocalendar()[['year', 'week']]
    
    weekends = pd.concat([weekends, weekends_iso], axis=1)
    weekdays = pd.concat([weekdays, weekdays_iso], axis=1)
    
    weekends_grp = weekends.groupby(['year', 'week']).agg({
        'comment_count': 'mean',
        'date': 'first'
    }).reset_index().round(2)
    
    weekdays_grp = weekdays.groupby(['year', 'week']).agg({
        'comment_count': 'mean',
        'date': 'first'
    }).reset_index().round(2)

    weekly_weekday_normality_p = stats.normaltest(weekdays_grp['comment_count']).pvalue
    weekly_weekend_normality_p = stats.normaltest(weekends_grp['comment_count']).pvalue
    weekly_levene_p = stats.levene(weekends_grp['comment_count'],weekdays_grp['comment_count']).pvalue  
    weekly_ttest_p = stats.ttest_ind(weekends_grp['comment_count'],weekdays_grp['comment_count']).pvalue

    #u test
    
    from scipy.stats import mannwhitneyu
    
    U1, p = mannwhitneyu(weekends['comment_count'], weekdays['comment_count'],alternative='two-sided')
    
    utest_p=p
    

    print(OUTPUT_TEMPLATE.format(
        
        initial_ttest_p = initial_ttest_p,
        initial_weekday_normality_p = initial_weekday_normality_p,
        initial_weekend_normality_p = initial_weekend_normality_p,
        initial_levene_p = initial_levene_p,
        transformed_weekday_normality_p = transformed_weekday_normality_p,
        transformed_weekend_normality_p = transformed_weekend_normality_p,
        transformed_levene_p = transformed_levene_p,
        weekly_weekday_normality_p = weekly_weekday_normality_p,
        weekly_weekend_normality_p = weekly_weekend_normality_p,
        weekly_levene_p = weekly_levene_p,
        weekly_ttest_p = weekly_ttest_p,
        utest_p = utest_p,

        
    ))
    
if __name__ == '__main__':
    main()

