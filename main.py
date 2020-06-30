from snownlp import SnowNLP
import jieba
# from mySnownlp import SnowNLP
import pandas as pd
import openpyxl

# text = '这个东西不是一般好'
s=SnowNLP('差得要死')
print(s.sentiments)


# f=pd.read_excel("2_organize.xlsx",names=['text']);


# new_table=f


###predict tag
# new_table['tag']=new_table.apply(lambda x:1 if SnowNLP(x['text']).sentiments>0.5 else 0,axis=1)
# print(new_table)
# new_table.to_excel('out1_tag.xlsx')


###tokenize
# print(len(f))
# print(type(f))
# new_table['token']=new_table.apply(lambda x:SnowNLP(x['text']).words,axis=1)
# print(new_table)
# new_table.to_excel('out2_token.xlsx')


# new_table['score']=new_table.apply(lambda x:SnowNLP(x['text']).sentiments,axis=1)
# print(new_table)
# new_table.to_excel('out3_score.xlsx')