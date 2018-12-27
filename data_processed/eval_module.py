import pandas as pd
path_pred='../outs/test_results.tsv'
path_label='../data_examples/dev.tsv'
import numpy as np
df_text=pd.read_csv(path_label,delimiter='\t',header=None).rename(columns={0:'label',1:'text'})


df_pred=pd.read_csv(path_pred,header=None).rename(columns={0:'p'})
print(df_pred.head())

def get_pred_label(input_string):
    print(input_string)
    p_l=np.float32(np.array(input_string.split('\t')))
    pred_label = np.argmax(p_l, 0)
    return pred_label

df_pred['pred_label']=df_pred.apply(lambda row:get_pred_label(row['p']),axis=1)
print(df_pred.head())

df_all=pd.concat([df_text,df_pred['pred_label']],axis=1)
df_all['judge']=(df_all['label']==df_pred['pred_label']).astype(int)
print('acc',df_all['judge'].mean())
print(df_all.columns,df_all.head())
