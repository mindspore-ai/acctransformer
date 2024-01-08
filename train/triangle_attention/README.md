# TriangleAttention介绍

## 一、简介

acctransformer中TriangleAttention是针对自回归类大模型的Attention模块存在无效计算和无效数据的优化。
自回归类大模型的Attention模块采用了一个attention mask来使得前面的token不会attention到后面的token，
这部分被mask掉的数据和对应的计算对Attention模块的输出均无意义。
因此TriangleAttention将采用分块策略来过滤掉这些无效数据和无效计算，以提升Attention的计算效率和自回归类模型的端到端性能。

## 二、安装使用

### 2.1、环境安装

#### 2.1.1、配套环境要求

MindSpore: [2.2.0](https://www.mindspore.cn/versions#2.2.0) <br>
MindSpore官方网站：[链接](https://www.mindspore.cn/install) <br>

CANN配套软件包版本以及安装参考MindSpore安装文档内教程。

#### 2.1.2、安装

安装TriangleAttention：

1. 直接克隆源码使用，使用源码方式调用时设置PYTHONPATH

```bash
export PYTHONPATH=/yourcodepath/acctransformer/train:$PYTHONPATH
```

2. 安装whl包使用（当前Mindspore2.2.0版本从系统路径调用有问题，暂不使用该安装方式）

```bash
   cd train
   python setup.py install
```

或者

```bash
   cd train
   bash build.sh
   pip install dist/acctransformer-1.0.0-py3-none-any.whl
```

#### 2.1.3、注意事项

1. 只有自回归类AI模型可用，attention_mask如下所示：

```python
attention_mask = Tensor(np.triu(np.ones((batch_size, seq_length, seq_length), dtype=np.float16), k=1))
```

2. triangle_attention里的attention score不会除以sqrt(head_dim)， 调用者需要事先做query=query/sqrt(head_dim)，或key=key/sqrt(head_dim)，或者query=query/sqrt(sqrt(head_dim))且key=key/sqrt(sqrt(head_dim))

3. 训练文本较短，如seq_length小于2048时，该特性做数据切分的开销会抵消掉减少attention无效计算带来的收益，性能可能会劣化。因此建议在训练文本较长，如seq_length大于等于2048时使用该特性。

#### 2.1.3、建议事项

1. block_size建设设置成seq_length的1/4性能最佳。

### 2.2、TriangleAttention使用方法

#### 使用接口

```train/triangle_attention/triangle_attention.py```

**接口简介**

```python

class TriangleAttention(nn.Cell):
    """Triangle Attention Layer.

    This function contains the triangle attention primitives.
    The triangle attention divides the q, k and v into blocks to eliminate invalid calculations and invalid data in the
    mask part of attention score.

    Specifically, it includes the following:

    1. An interface for calling triangle operation.
    2. A configuration parameter for adjusting block size.

    Args:
        block_size(int): An integer determining the block size.
            Default 512.
        dropout_rate(float): The dropout rate of the attention score.
            Default 0.0.
        dp(int): data parallel.
            Default 1.
        mp(int): model parallel.
            Default 1.


    Inputs:
      - **query** (Tensor) - Tensor query (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **key** (Tensor) - Tensor key (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **value** (Tensor) - Tensor value (:class:`mstype.fp16` [batch_size, head_num, seq_length, head_dim])
      - **attention_mask** (Tensor) - Float Tensor the mask of (:class:`mstype.fp16` [batch_size, seq_length,
          seq_length]): A matrix to pass masked information.

    Outputs:
        A Tensor. The output of the attention with shape [batch_size, head_num, seq_length, head_dim]

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> from train.triangle_attention.triangle_attention import TriangleAttention
        >>> model = TriangleAttention(block_size=1024)
        >>> query = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> key = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> value = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> attention_mask = Tensor(np.triu(np.ones((2, 4096, 4096)), k=1), mstype.float16)
        >>> output = model(query, key, value, attention_mask)
        >>> print(output.shape)
        (2, 16, 4096, 128)
    """

```

## 三、分支以及版本说明

初始版本，后续待补充

## 四、性能比较

初始版本，后续待补充

## 五、测试

* ut测试：进入tests/train/triangle_attention/ut目录

全量测试运行 `pytest -sv .` ; 测试单个文件或函数用例`pytest -sv test_XXX.py::test_func_name`即可