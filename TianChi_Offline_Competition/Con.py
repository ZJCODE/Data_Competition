import glob
import pandas as pd
import numpy as np
Results = glob.glob(r'*.csv')

Buy1 = pd.read_csv(Results[0],sep='\t')
Buy2 = pd.read_csv(Results[1],sep='\t')


print len(Buy1),len(Buy2)

Buy = pd.merge(Buy1,Buy2)

print len(Buy)

def save(df):
	df.to_csv('tianchi_mobile_recommendation_predict.csv', sep='\t', columns=['user_id','item_id'], index=False, encoding='utf-8')

save(Buy)
