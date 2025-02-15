import pandas as pd
from scipy.stats import chi2_contingency
from scipy import stats
import sys

data = pd.read_json(sys.argv[1], orient='records', lines=True)

data

#analysis for instructors only 
#searchdata_file=searchdata_file[searchdata_file['is_instructor']==True]
#searchdata_file

#searchdata_file

def test(searchdata_file):
    #old even
    old=searchdata_file[searchdata_file['uid']%2==0]
    
    #new odd
    new=searchdata_file[searchdata_file['uid']%2==1]
    
    
    #old interface
    even_searched_atleast_once=searchdata_file[(searchdata_file['search_count']>0 )& (searchdata_file['uid']%2==0)].shape[0]
    even_never_searched=searchdata_file[(searchdata_file['search_count']==0 )& (searchdata_file['uid']%2==0)].shape[0]
    
    #new
    odd_searched_atleast_once=searchdata_file[(searchdata_file['search_count']>0) & (searchdata_file['uid']%2==1)].shape[0]
    
    odd_never_searched=searchdata_file[(searchdata_file['search_count']==0) & (searchdata_file['uid']%2==1)].shape[0]
    
    
    #contingency table
    contingency_table = [[even_searched_atleast_once,even_never_searched],[odd_searched_atleast_once,odd_never_searched]]
    
    # chisquared test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # mannwhitneyu test
    stat, p_man = stats.mannwhitneyu(new['search_count'], old['search_count'], alternative= 'two-sided')
    
    return (p_value,p_man)

OUTPUT_TEMPLATE = (
    'Did more/less users use the search feature? p-value:  {more_users_p:.3g}\n'
    'Did users search more/less? p-value:  {more_searches_p:.3g} \n'
    'Did more/less instructors use the search feature? p-value:  {more_instr_p:.3g}\n'
    'Did instructors search more/less? p-value:  {more_instr_searches_p:.3g}' 
)

#analysis for instructors only 
instr_data=data[data['is_instructor']==True]

more_users_p,more_searches_p= test(data)
more_instr_p, more_instr_searches_p=test(instr_data)
print(OUTPUT_TEMPLATE.format(more_users_p=more_users_p, more_searches_p=more_searches_p,more_instr_p=more_instr_p,more_instr_searches_p=more_instr_searches_p))

