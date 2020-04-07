import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']#把图表内的文字转换成中文
plt.rcParams['axes.unicode_minus']=False
from scipy import stats
import os
%matplotlib inline
from sklearn.linear_model import LinearRegression#导入线性回归模块

def datasum():
    path ='D:/data analysis/py/example/'
    folder =os.walk(path)#打开路径下的文件夹
    files =list(folder)[0][2]   #遍历文件
    n=1#第一个数据
    for i in files:
        file =path +i#路径下+数据
        data =pd.read_excel(file,index_col=0)#重新设置indexx的值
        data_counts=len(data)
        columns =data.columns.tolist()#将矩阵或数组转换成列表
        nan_counts =len(data[data.isnull().values==True])
        print('第{}个数据数据量为：{}'.format(n,data_counts))
        print('第{}个数据数据字段为:{}'.format(n,columns))
        print('第{}个数据数据量缺失值为：{}'.format(n,nan_counts))
        print('---------')
        n+=1
        
datasum()        
print('finished!')    

def fill():
    path ='D:/data analysis/py/example/'
    folder =os.walk(path)
    files =list(folder)[0][2]   
    data_files=[]
    for i in files:
        file =path +i
        data =pd.read_excel(file,index_col=0)
        columns =data.columns.tolist()
        data.to_period()#对日期字段进行时间序列处理
        data[columns[0]].fillna(data[columns[0]].mean(),inplace =True)
        data[columns[1]].fillna(data[columns[1]].mean(),inplace =True)#用均值填充缺失值
        data_files.append(data)
    return(data_files) 
    
data1, data2, data3 = fill()[0], fill()[1], fill()[2]
def sum_p(*data_files):
    path ='D:/data analysis/py/example/'
    A_sale=[]   
    B_sale=[]
    for data in data_files:
        columns=data.columns
        A_sale.append(data[columns[0]].sum())#计算A产品月总销量
        B_sale.append(data[columns[1]].sum())
    df =pd.DataFrame({'A_sale_sum':A_sale,'B_sale_sum':B_sale},
                     index =pd.period_range('201801','201803',freq='M')) 
    print(df)
    plt.figure()
    df.plot(kind = 'bar',color = ['r','k'],alpha = 0.5, rot = 0,figsize = (8,4))
    plt.title('1-3月A,B产品总销量',color ='r')
    plt.ylim([15000,25000])
    plt.legend(loc = 'upper left')
    plt.grid()
    plt.savefig('D:/data analysis/py/new/' + '1-3月A,B产品总销量.png',dpi=400)
sum_p(data1,data2,data3)       


def pre_sum(*data_files):
    keydates = []
    for data in data_files:
        columns = data.columns  # 提取列名
        data['A_sale_sum%'] = data[columns[0]].cumsum() / data[columns[0]].sum()  # 计算A产品累计销量占比
        keydate = data[data['A_sale_sum%']>0.8].index[0]  
        keydates.append(str(keydate))
        # 记录销量超过80%的日期
    print('A产品月度超过80%的销量日期分别为\n', keydates)
    return(keydates)
pre_sum(data1,data2,data3)

def delnan():
    path = 'D:/data analysis/py/example/'
    folder = os.walk(path)   # 遍历文件
    files = list(folder)[0][2]
    data_files = []
    for i in files:
        file = path + i
        data = pd.read_excel(file,index_col = 0)
        columns = data.columns.tolist()  # 提取列名
        data.to_period()  # 转换成时间序列
        data.dropna(inplace=True)  # 删除缺失值
        data_files.append(data)
    data = pd.concat([data_files[0],data_files[1],data_files[2]])  # 合并数据
    return(data)
    
data = delnan()
def Linear(data):
    path = 'D:/data analysis/py/example/'
    model = LinearRegression()
    model.fit(data['productA'][:,np.newaxis],data['productB'])  #样本数据参数录入，列变量
    # 构建回归模型
    xtest = np.linspace(0,1000,1000)#特征值
    ytest = model.predict(xtest[:,np.newaxis])#做预测
    plt.scatter(data['productA'],data['productB'],marker = '.',color = 'grey')
    plt.plot(xtest,ytest,color = 'r')
    plt.grid(True)
    plt.title('A-B产品销量回归拟合')
    plt.savefig('D:/data analysis/py/new/' + 'A-B产品销量回归拟合.png',dpi=400)  
    # 存储图表
    a=[[1200],[2000]]
    return(model.predict(a))
print(Linear(data))
