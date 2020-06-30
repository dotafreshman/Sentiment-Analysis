from snownlp import SnowNLP
import pandas as pd
import openpyxl
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

f=pd.read_excel("acc.xlsx",names=['pred', 'true']);


new_table=f


# new_table['tag']=new_table.apply(lambda x:1 if SnowNLP(x['text']).sentiments>0.5 else 0,axis=1)
# print(new_table)
# new_table.to_excel('out1_tag.xlsx')
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')




ax=plt.subplots()
y_true = new_table['pred']
y_pred = new_table['true']
# print(y_true)
C2= confusion_matrix(y_true, y_pred, labels=[0,1])
print(C2)
sns.heatmap(C2,annot=True)




