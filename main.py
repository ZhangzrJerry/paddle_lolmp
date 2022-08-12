import pandas as pd
import numpy as np
import paddle
# 导入训练数据
df_train = pd.read_csv("train.csv")
x = np.asarray(df_train.iloc[:,2:]).astype(np.float32)
y = np.array(df_train.iloc[:,1]).astype(np.int8)
# 导入测试数据
df_test = pd.read_csv("test.csv")
x_pred = np.array(df_test.iloc[:,1:])
# 对特征进行归一化
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(x)  
x = scaler.transform(x)
x_pred = scaler.transform(x_pred)
# 将训练数据集和测试数据集按照8:2的比例分开
ratio = 0.8
offset = int(x.shape[0] * ratio)
x_train = x[:offset]
y_train = y[:offset]
x_test = x[offset:]
y_test = y[offset:]

import numpy as np
from paddle.io import Dataset
class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self, x, y):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super(MyDataset, self).__init__()
        self.data_list = []
        for i,j in zip(x,y):
            self.data_list.append([i,j])

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        data = self.data_list[index]
        feature = np.array(data[:-1]).astype('float32')
        label = np.array(data[-1:]).astype('int64')
        # 返回特征和对应标签
        return feature, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

train_dataset = MyDataset(x_train,y_train)
test_dataset = MyDataset(x_test,y_test)
train_dataset[0]

n_input = 30
# MLP模型组网搭建
from paddle import nn
lenet_Sequential = nn.Sequential(
    nn.Linear(n_input, 1,)
)
# paddle.device.set_device('gpu:0')  # 本地显卡MX150没装CUDA
# 封装模型为一个 model 实例，便于进行后续的训练、评估和推理
model = paddle.Model(lenet_Sequential)
# 为模型训练做准备，设置优化器及其学习率，并将网络的参数传入优化器，设置损失函数和精度计算方式
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()), 
              loss=paddle.nn.MSELoss(), 
              metrics=paddle.metric.Accuracy())
# 启动模型训练，指定训练数据集，设置训练轮次，设置每次数据集计算的批次大小，设置日志格式
model.fit(train_dataset, 
          test_dataset,
          epochs=1, 
          batch_size=1,
          verbose=1)