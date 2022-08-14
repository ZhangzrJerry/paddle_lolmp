#### 20220812
这里有个bug的点，paddle框架自带的波士顿房价dataset和我自己的dataset，`__item__`方法返回的都是一个元组，用`a,b = train_dataset=[0]`的话那就是两个np数组
```py
# 自建dataset逐行输出结果
(array([-1.0423225 , -0.24851945, -1.0652921 , -1.0532238 , -0.43834025,
        -0.19826205, -0.58423096, -0.24698801, -0.09784286, -0.04000573,
        -1.188272  , -0.7809258 , -0.6200115 , -0.28191438, -0.5774179 ,
...
        -1.2297674 , -0.8086368 , -0.564268  , -0.61155623, -0.82438904,
        -0.10260557, -0.7286519 ,  0.        , -1.2983824 , -1.155188  ,
        -1.1353154 , -0.51228553, -0.9952254 , -0.35158026, -0.33337447],
       dtype=float32),
 array(0, dtype=int64))
 ```
 ```py
 # paddle自带dataset逐行输出结果
(array([-0.0405441 ,  0.06636364, -0.32356227, -0.06916996, -0.03435197,
         0.05563625, -0.03475696,  0.02682186, -0.37171335, -0.21419304,
        -0.33569506,  0.10143217, -0.21172912]),
 array([24.]))
 ```
我寻思他们也没差别，但问题很明显是锁定在dataset上面，下面表格应该很好地反应出来，相关度拉满好吧。
||自建dataset|飞桨dataset|
|-|-|-|
|**自建网络**|×|✔|
|**示例网络**|×|✔|

自己的dataset扔进去就愣报错好吧
```
ValueError: (InvalidArgument) The type of data we are trying to retrieve does not match the type of data currently contained in the container. 
[Hint: Expected y_dims[y_ndim - 2] == K, but received y_dims[y_ndim - 2]:30 != K:13.] (at C:\home\workspace\Paddle_release\paddle/phi/kernels/impl/matmul_kernel_impl.h:315)
[operator < matmul_v2 > error]
```
不过我又发现，如果n_input的值没有和输入特征对应的话也会出现error，于是我锁定了以下这段代码
```py
n_input = len(x[0])
# MLP模型组网搭建
from paddle import nn
lenet_Sequential = nn.Sequential(
    nn.Linear(n_input, 1,)
)
```
我以为是`len(x[0])`的问题，于是直接让`n_input=30`但还是报错，重新看发现两次报的其实错误不同，指并不是n_input不匹配的问题
他说我传入的数据类型不匹配，但自建的dataset和飞桨的dataset读出来的数据类型其实也没那么多不同，这让我非常困惑。于是我去看了UCIHousing的源代码
```py
def __getitem__(self, idx):
    data = self.data[idx]
    return np.array(data[:-1]).astype(self.dtype), \
            np.array(data[-1:]).astype(self.dtype)
```
而设置`self.data`部分的代码在`_load_data`方法中
```py
def _load_data(self, feature_num=14, ratio=0.8):
    data = np.fromfile(self.data_file, sep=' ')
    data = data.reshape(data.shape[0] // feature_num, feature_num)
    maximums, minimums, avgs = data.max(axis=0), data.min(axis=0), data.sum(
        axis=0) / data.shape[0]
    for i in six.moves.range(feature_num - 1):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
    offset = int(data.shape[0] * ratio)
    if self.mode == 'train':
        self.data = data[:offset]
    elif self.mode == 'test':
        self.data = data[offset:]
```
前两步的话知识把data从保存的文件中读出然后变形，在UCIHousing中data应该是一个一维张量，第二行应该是在做切分（13个特征+1个标签），接下来的两行是通过一个二元函数对特征值进行放缩，然后offset开始切分数据集为训练集和测试集
那我明白了，`__getitem__`这里其实也就是读取每一行，然后拆分出feature和label，但无论怎么样我的dataset和飞桨的dataset在`type()`下的输出结果还是相同的，一个是`tuple` `numpy.ndarray`，另一个是`<class 'tuple'>` `<class 'numpy.ndarray'>`
我就不明白了，这两个dataset已经像的不能再像了，怎么还是不行，盖亚啊啊啊！！！！！！！！！！！！！！！！！！！！！！
你说咋的，当我把上面这段代码改为下面这段之后，模型训练那步能跑了，当我开心的呀看着他转了好久还不出来，心想大事不妙
```py
# 修改前
def __getitem__(self, index):
    feature = np.array(self.data_list[index][:-1]).astype('float32')
    label = np.array(self.data_list[index][-1:]).astype('int64')
    # 返回特征和对应标签
    return feature, label
```
```py
# 修改后
def __getitem__(self, index):
    data = self.data_list[index]
    feature = np.array(data[:-1]).astype('float32')
    label = np.array(data[index][-1:]).astype('int64')
    # 返回特征和对应标签
    return feature, label
```
果然报错了`IndexError: list index out of range`
那我又去看了`model.fit`的源代码，想看一下他到底是怎么样访问的
我用这段代码对HCIHousing的dataset进行访问是没有问题的，非常OK
```py
for i in range(len(train_dataset)):
    print(train_dataset.__getitem__(i)[1])
```
但对于自个的dataset而言果不其然地报错
```
{
	"name": "IndexError",
	"message": "invalid index to scalar variable.",
	"stack": "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m\n\u001b[1;31mIndexError\u001b[0m                                Traceback (most recent call last)\n\u001b[1;32md:\\项目\\让我们荡起飞桨\\飞桨_英雄联盟\\main.ipynb Cell 5\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      <a href='vscode-notebook-cell:/d%3A/%E9%A1%B9%E7%9B%AE/%E8%AE%A9%E6%88%91%E4%BB%AC%E8%8D%A1%E8%B5%B7%E9%A3%9E%E6%A1%A8/%E9%A3%9E%E6%A1%A8_%E8%8B%B1%E9%9B%84%E8%81%94%E7%9B%9F/main.ipynb#X11sZmlsZQ%3D%3D?line=0'>1</a>\u001b[0m \u001b[39mfor\u001b[39;00m i \u001b[39min\u001b[39;00m \u001b[39mrange\u001b[39m(\u001b[39mlen\u001b[39m(train_dataset)):\n\u001b[1;32m----> <a href='vscode-notebook-cell:/d%3A/%E9%A1%B9%E7%9B%AE/%E8%AE%A9%E6%88%91%E4%BB%AC%E8%8D%A1%E8%B5%B7%E9%A3%9E%E6%A1%A8/%E9%A3%9E%E6%A1%A8_%E8%8B%B1%E9%9B%84%E8%81%94%E7%9B%9F/main.ipynb#X11sZmlsZQ%3D%3D?line=1'>2</a>\u001b[0m     \u001b[39mprint\u001b[39m(train_dataset\u001b[39m.\u001b[39;49m\u001b[39m__getitem__\u001b[39;49m(i)[\u001b[39m1\u001b[39m])\n\n\u001b[1;32md:\\项目\\让我们荡起飞桨\\飞桨_英雄联盟\\main.ipynb Cell 5\u001b[0m in \u001b[0;36mMyDataset.__getitem__\u001b[1;34m(self, index)\u001b[0m\n\u001b[0;32m     <a href='vscode-notebook-cell:/d%3A/%E9%A1%B9%E7%9B%AE/%E8%AE%A9%E6%88%91%E4%BB%AC%E8%8D%A1%E8%B5%B7%E9%A3%9E%E6%A1%A8/%E9%A3%9E%E6%A1%A8_%E8%8B%B1%E9%9B%84%E8%81%94%E7%9B%9F/main.ipynb#X11sZmlsZQ%3D%3D?line=21'>22</a>\u001b[0m data \u001b[39m=\u001b[39m \u001b[39mself\u001b[39m\u001b[39m.\u001b[39mdata_list[index]\n\u001b[0;32m     <a href='vscode-notebook-cell:/d%3A/%E9%A1%B9%E7%9B%AE/%E8%AE%A9%E6%88%91%E4%BB%AC%E8%8D%A1%E8%B5%B7%E9%A3%9E%E6%A1%A8/%E9%A3%9E%E6%A1%A8_%E8%8B%B1%E9%9B%84%E8%81%94%E7%9B%9F/main.ipynb#X11sZmlsZQ%3D%3D?line=22'>23</a>\u001b[0m feature \u001b[39m=\u001b[39m np\u001b[39m.\u001b[39marray(data[:\u001b[39m-\u001b[39m\u001b[39m1\u001b[39m])\u001b[39m.\u001b[39mastype(\u001b[39m'\u001b[39m\u001b[39mfloat32\u001b[39m\u001b[39m'\u001b[39m)\n\u001b[1;32m---> <a href='vscode-notebook-cell:/d%3A/%E9%A1%B9%E7%9B%AE/%E8%AE%A9%E6%88%91%E4%BB%AC%E8%8D%A1%E8%B5%B7%E9%A3%9E%E6%A1%A8/%E9%A3%9E%E6%A1%A8_%E8%8B%B1%E9%9B%84%E8%81%94%E7%9B%9F/main.ipynb#X11sZmlsZQ%3D%3D?line=23'>24</a>\u001b[0m label \u001b[39m=\u001b[39m np\u001b[39m.\u001b[39marray(data[index][\u001b[39m-\u001b[39;49m\u001b[39m1\u001b[39;49m:])\u001b[39m.\u001b[39mastype(\u001b[39m'\u001b[39m\u001b[39mint64\u001b[39m\u001b[39m'\u001b[39m)\n\u001b[0;32m     <a href='vscode-notebook-cell:/d%3A/%E9%A1%B9%E7%9B%AE/%E8%AE%A9%E6%88%91%E4%BB%AC%E8%8D%A1%E8%B5%B7%E9%A3%9E%E6%A1%A8/%E9%A3%9E%E6%A1%A8_%E8%8B%B1%E9%9B%84%E8%81%94%E7%9B%9F/main.ipynb#X11sZmlsZQ%3D%3D?line=24'>25</a>\u001b[0m \u001b[39m# 返回特征和对应标签\u001b[39;00m\n\u001b[0;32m     <a href='vscode-notebook-cell:/d%3A/%E9%A1%B9%E7%9B%AE/%E8%AE%A9%E6%88%91%E4%BB%AC%E8%8D%A1%E8%B5%B7%E9%A3%9E%E6%A1%A8/%E9%A3%9E%E6%A1%A8_%E8%8B%B1%E9%9B%84%E8%81%94%E7%9B%9F/main.ipynb#X11sZmlsZQ%3D%3D?line=25'>26</a>\u001b[0m \u001b[39mreturn\u001b[39;00m feature, label\n\n\u001b[1;31mIndexError\u001b[0m: invalid index to scalar variable."
}
```
还有逗的啊，我的dataset跑上面的代码能行，跑下面的不行
```py
for i in range(len(train_dataset)):
    print(train_dataset.data_list[i][1])
```
但是飞桨自带的dataset就可以
```py
for i in range(len(train_dataset)):
    print(train_dataset.__getitem__(i)[1])
```
我自己那个dataset只打印第一行的时候又可以`print(train_dataset.__getitem__(0))`我真的不理解呀，第二行又不行了
我理解了，我代码扣错了，给label赋值那行多打了个index索引进去
```py
def __getitem__(self, index):
    data = self.data_list[index]
    feature = np.array(data[:-1]).astype('float32')
    label = np.array(data[index][-1:]).astype('int64')
    # 返回特征和对应标签
    return feature, label
```
我不理解哈哈哈咋还有问题，还是之前那个Invalid Value问题
又发现了一个小问题，下面这段代码打印出来的是label
```py
for i in range(len(train_dataset)):
    print(train_dataset.__getitem__(i)[0])
```
这样子就能正确打印出feature和label
```py
for i in range(len(train_dataset)):
    a,b = train_dataset.__getitem__(i)
    print(a,b)
```
我擦擦擦VSC崩了，我的.ipynb代码块全部空了出来，好在恢复过来了，还有一个问题是他丫的.ipynb中一调试就退出，真不好玩，我又不想用pycharm这么重量的
后来在VSC里建了一个.py文件来debug，或者叫他比较有意思的事情吧，就是我的debug第一次通过`model.fit`的时候是没问题的，第二次通过的时候就抛出异常了
只是有时候气呀，明明能用高层API解决的问题，为啥要去手写底层代码
稍微看了下model.py里好像也要先把dataset转为dataloader，说实话感觉sklearn和keras做这种任务的时候让人更加轻松，直接扔numpy张量就行，或者paddle好像也没有能够直接把数据转为paddle格式的tensor，需要自己来自定义，也不是不可以吧其实说，但是的话先预处理完feature再转换tensor也不是不可以，现在主要的问题是我没有头绪
emm好像有个to_tensor方法，但是如果想要直接调用高层API`model.fit`的话还是不行呀
奈斯奈斯我爽了，解决了归一化的问题，利用了`np.c_[]`方法，终于正常地开始训练了哈哈哈哈哈哈，不多说了直接看代码吧
```py
# 导入训练数据
df_train = pd.read_csv("train.csv")
df_train = df_train.drop(['id'],axis=1)
# 对特征进行归一化
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(df_train.iloc[:,0:])  
scaler_data = scaler.transform(df_train.iloc[:,0:])
# 将训练数据集和测试数据集按照8:2的比例分开
ratio = 0.8
offset = int(df_train.shape[0] * ratio)
train_data = np.c_[scaler_data,df_train.iloc[:,0]][:offset].copy()
test_data = np.c_[scaler_data,df_train.iloc[:,0]][offset:].copy()
# MLP模型组网搭建
n_input = 30
from paddle import nn
class Classifier(paddle.nn.Layer):
    def __init__(self):
        super(Classifier, self).__init__()
        self.l1 = paddle.nn.Linear(n_input, 1,)

    def forward(self, inputs):
        pred = self.l1(inputs)
        return pred
# 训练过程
import paddle.nn.functional as F 
y_preds = []
train_nums = []
train_costs = []
labels_list = []
BATCH_SIZE = 20

def train(model):
    print('start training ... ')
    # 开启模型训练模式
    model.train()
    EPOCH_NUM = 5
    train_num = 0
    optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(train_data)
        # 将训练数据进行拆分，每个batch包含20条数据
        mini_batches = [train_data[k: k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
        for batch_id, data in enumerate(mini_batches):
            features_np = np.array(data[:, :n_input], np.float32)
            labels_np = np.array(data[:, -1:], np.float32)
            features = paddle.to_tensor(features_np)
            labels = paddle.to_tensor(labels_np)
            # 前向计算
            y_pred = model(features)
            cost = F.mse_loss(y_pred, label=labels)
            train_cost = cost.numpy()[0]
            # 反向传播
            cost.backward()
            # 最小化loss，更新参数
            optimizer.step()
            # 清除梯度
            optimizer.clear_grad()
            
            if batch_id%30 == 0 and epoch_id%50 == 0:
                print("Pass:%d,Cost:%0.5f"%(epoch_id, train_cost))

            train_num = train_num + BATCH_SIZE
            train_nums.append(train_num)
            train_costs.append(train_cost)
        
model = Classifier()
train(model)

# 损失函数曲线
def draw_train_process(iters, train_costs):
    plt.title("training cost", fontsize=24)
    plt.xlabel("iter", fontsize=14)
    plt.ylabel("cost", fontsize=14)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.show()

import matplotlib
matplotlib.use('TkAgg')
%matplotlib inline
draw_train_process(train_nums, train_costs)
```
当然还有一些细节没有细扣，先睡觉了，明天继续
***
#### 20220813
那这部分作为预测部分的代码
```py
# 获取预测数据
INFER_BATCH_SIZE = 100

infer_features_np = np.array(test_data[1:]).astype("float32")
infer_labels_np = np.array(test_data[0]).astype("float32")

infer_features = paddle.to_tensor(infer_features_np)
infer_labels = paddle.to_tensor(infer_labels_np)
fetch_list = model(infer_features)

sum_cost = 0
for i in range(INFER_BATCH_SIZE):
    infer_result = fetch_list[i][0]
    ground_truth = infer_labels[i]
    if i % 10 == 0:
        print("No.%d: infer result is %.2f,ground truth is %.2f" % (i, infer_result, ground_truth))
    cost = paddle.pow(infer_result - ground_truth, 2)
    sum_cost += cost
mean_loss = sum_cost / INFER_BATCH_SIZE
print("Mean loss is:", mean_loss.numpy())
```
利用debug功能，定位到了错误代码，是在类`Classifier`中的`forward`方法
```py
def forward(self, inputs):
    pred = self.l1(inputs)
```
报错内容为
```
发生异常: ValueError
(InvalidArgument) Input(Y) has error dim.Y'dims[0] must be equal to 32But received Y'dims[0] is 30
  [Hint: Expected y_dims[y_ndim - 2] == K, but received y_dims[y_ndim - 2]:30 != K:32.] (at C:\home\workspace\Paddle_release\paddle/phi/kernels/impl/matmul_kernel_impl.h:315)
  [operator < matmul_v2 > error]
```
很快我们定位到这两行代码，当然除此以外还有一些小补丁，问题比较容易排查就没列在这里
```py
infer_features_np = np.array(test_data[1:]).astype("float32")
infer_labels_np = np.array(test_data[0]).astype("float32")
```
修改切片方式之后又有其他地方报错，是在train部分的前向计算中
```py
cost = F.binary_cross_entropy_with_logits(y_pred, label=labels)
```
```
发生异常: ValueError
(InvalidArgument) Input(X) and Input(Label) shall have the same rank.But received: the rank of Input(X) is [2], the rank of Input(Label) is [1].
  [Hint: Expected rank == labels_dims.size(), but received rank:2 != labels_dims.size():1.] (at C:\home\workspace\Paddle_release\paddle\phi\infermeta\binary.cc:1735)
  [operator < sigmoid_cross_entropy_with_logits > error]
```
于是我把08.12已经成功的训练部分代码挪回来，把一下错误的细节修改，例如提取feature和label的部分，那放进文件里可以跑通训练部分，但当我把lossfunction从mseloss修改为binary_cross_entropy_with_logits之后就炸了，但也有个很有意思的地方是我之前跑交叉熵损失函数时是有成功过的，这就让人比较抓狂了
其实不需要挪用之前的代码，emm报错很明显提示了两个张量的秩有差异，从debug的位置来看一个是(20,1)另一个是(20)，于是我在进行loss计算前用reshape方法对y_pred进行降维操作，又可以继续训练了，虽然训练到一半窗口未响应remake了
```py
y_pred = paddle.reshape(y_pred,shape=[-1])
```
> 有一点我需要注意的是，我应该先看清楚报错的问题是什么再去解决，它都已经把异常抛出了，就没必要说去尝试到底是哪里出现的问题
也是比较骚啊，这里交叉熵损失函数出现了反复波动，而且取值小于0的情况，参考 https://blog.csdn.net/qq_39575835/article/details/104353889 ，在输出层加入激活函数softmax，报错啦
### 20220814
后面的问题都比较小了，今天早上在仓库提交了新的代码，训练和预测都能够运行了。现在遇到了一个大问题，就是执行完sigmoid后，预测结果会有负数，这是最骚的