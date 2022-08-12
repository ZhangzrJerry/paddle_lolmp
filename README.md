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