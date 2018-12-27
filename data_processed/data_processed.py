import pandas as pd
import os
from sklearn.model_selection import train_test_split
def data_processed(path_read,path_save):
    all_data_list=[]
    '''先分词将一些实体替换掉，例如日期时间、人名、地名、机构名字,还是先用规则处理'''
    # 1.去除特殊字符，直接去处所有的标点符号
    path1=os.path.join(path_read,'neg.txt')#负向label==1
    path2=os.path.join(path_read,'pos.txt')#正向label==0
    with open(path1,encoding='gbk',errors='ignore') as neg_file:
        for each in neg_file:
            each=each.strip()
            each_line_dict={}
            
            each_line_dict['text']=each
            each_line_dict['label'] = 1
            all_data_list.append(each_line_dict)
    with open(path2,encoding='gbk',errors='ignore') as neg_file:
        for each in neg_file:
            each = each.strip()
            each_line_dict={}
            
            each_line_dict['text']=each
            each_line_dict['label'] = 0
            all_data_list.append(each_line_dict)

    df=pd.DataFrame(all_data_list)


    print('df.head',df.head())
    print('df.tail',df.tail())
    print('开始保存')
    # df[['text', 'label']].to_parquet(path_save, compression = 'gzip')
    X=df['text']
    y=df['label']

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, test_size=0.3, stratify=y, random_state=42)
    # print(
    #     '\n\n',
    #     X_train,
    #     '\n\n',
    #     y_train,
    #     '\n\n',
    #     X_test,
    #     '\n\n',
    #     y_test,
    # )
    df7=pd.DataFrame()
    df7['label'] = y_train
    df7['text']=X_train


    df3=pd.DataFrame()
    df3['label'] = y_test
    df3['text']=X_test


    df7.to_csv(os.path.join(path_save,'dev.tsv'),index=False,header=False,sep='\t')
    df3.to_csv(os.path.join(path_save,'train.tsv'), index=False, header=False, sep='\t')



if __name__=='__main__':
    path_read='hotel'
    path_save='../data_examples'

    data_processed(path_read, path_save)
    pass