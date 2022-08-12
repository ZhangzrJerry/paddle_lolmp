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
> ValueError: (InvalidArgument) The type of data we are trying to retrieve does not match the type of data currently contained in the container.
[Hint: Expected y_dims[y_ndim - 2] == K, but received y_dims[y_ndim - 2]:30 != K:13.] (at C:\home\workspace\Paddle_release\paddle/phi/kernels/impl/matmul_kernel_impl.h:315)
[operator < matmul_v2 > error]

不过我又发现，如果

```py
n_input = len(x[0])
# MLP模型组网搭建
from paddle import nn
lenet_Sequential = nn.Sequential(
    nn.Linear(n_input, 1,)
)
```