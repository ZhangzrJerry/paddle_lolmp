#### 20220801
* 定义一个计算汉明距离的函数
    ```python
    def hanming(a,b):
        res = 0
        for i,j in zip(a,b):
            res += (0 if i==j else 1)
        return res
    ```
* 导入数据集
    ```python
    # 导入训练数据
    df_train = pd.read_csv("train.csv")
    x = np.asarray(df_train.iloc[:,2:]).astype(np.float64)
    y = np.array(df_train.iloc[:,1])
    # 导入测试数据
    df_test = pd.read_csv("test.csv")
    x_pred = np.array(df_test.iloc[:,1:])
    # 把训练数据分为训练集和验证集
    x_train = x[:170000]
    y_train = y[:170000]
    x_test = x[170000:]
    y_test = y[170000:]
    ```