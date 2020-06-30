# from snownlp import SnowNLP
import jieba
from mySnownlp import SnowNLP
import pandas as pd
import openpyxl

# text = '这个东西不是一般好'



f=pd.read_excel("../a3_xlsx_file/kai-jieba-seg.xls");


new_table=f


# new_table['token']=new_table.apply(lambda x: jieba.lcut(x['seg']),axis=1)


# print(type(f['seg'][1].split(' ')))
###predict tag
new_table['score']=new_table.apply(lambda x:SnowNLP("一",x['seg'].split(' ')).mySent,axis=1)
new_table['pred']=new_table.apply(lambda x:1 if SnowNLP("一",x['seg'].split(' ')).mySent>0.5 else 0,axis=1)




new_table.to_excel('jiebaOut.xlsx')