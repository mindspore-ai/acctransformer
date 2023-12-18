# FlashAttention2介绍
## 一、简介
acctransformer中FlashAttention2算子基于昇腾达芬奇硬件和 CANN 软件栈进行开发，相较于传统的 attention 计算性能提升明显。

FlashAttention以及FlashAttention2算法参考以下论文：

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré  
Paper: https://arxiv.org/abs/2205.14135  
IEEE Spectrum [article](https://spectrum.ieee.org/mlperf-rankings-2022) about our submission to the MLPerf 2.0 benchmark using FlashAttention.

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**  
Tri Dao

Paper: https://tridao.me/publications/flash2/flash2.pdf


## 二、安装使用
### 2.1、环境安装
#### 2.1.1、配套环境要求
MindSpore: [2.2.0](https://www.mindspore.cn/versions#2.2.0) <br>
MindSpore官方网站：[链接](https://www.mindspore.cn/install) <br>

CANN配套软件包版本以及安装参考MindSpore安装文档内教程。

#### 2.1.2、安装

安装FlashAttention2：
1. 直接克隆源码使用，使用源码方式调用时设置PYTHONPATH
```bash
export PYTHONPATH=/yourcodepath/acctransformer/train:$PYTHONPATH
```
2. 安装whl包使用
```bash
   cd train
   python setup.py install
```
或者
```bash
   cd train
   bash build.sh
   pip install dist/flash_attention-23.12.11-py3-none-any.whl
```

#### 2.1.3、注意事项
1. 当前仅支持Ascend 910硬件


### 2.2、FlashAttention2使用方法

#### 使用接口
```train/flash_attention/nn/layer/flash_attention.py```

```python
from flash_attention.nn.layer.flash_attention import FlashAttention
```

#### 输入
1. 输入Q、K、V的 shape 支持：seq_length>=64 * 1024 (64K)；由于硬件限制，head_dim<=128。
2. 输入attention_mask的 shape 支持：(1,tiling_block_size,tiling_block_size)，tiling_block_size根据tiling策略不同变化，默认tiling_block_size为128，即默认attention_mask的shape为(1, 128, 128)；attention_mask的内容为全1的上三角矩阵，示例如下：
```python
import numpy as np
import mindspore as ms
from mindspore import Tensor

attention_mask = np.triu(np.ones((1, 128, 128), dtype=np.float16), k=1)
attention_mask = Tensor(attention_mask, dtype=ms.float16)
print(attention_mask)


[[[0. 1. 1. ... 1. 1. 1.]
  [0. 0. 1. ... 1. 1. 1.]
  [0. 0. 0. ... 1. 1. 1.]
  ...
  [0. 0. 0. ... 0. 1. 1.]
  [0. 0. 0. ... 0. 0. 1.]
  [0. 0. 0. ... 0. 0. 0.]]]
```


**接口简介**
```python
class FlashAttention(Cell):
    """Flash Attention Layer.

    This function contains the flash attention primitives used in FlashAttention (see paper)
    `FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning <https://tridao.me/publications/flash2/flash2.pdf>`

    Specifically, it includes the following:

    1. An interface for calling flashattention operation.
    2. Two configuration parameters for enabling local block sparse of flashattention.

    Args:
        head_dim(int): The hidden size of input.
        dropout_rate(float): The dropout rate of the attention score. Default 0.0.
        prev_block_num(int): A integer to define the number of blocks to look ahead for local block sparse attention.
            Default 65536.
        next_block_num(int): A integer to define the number of blocks to look behind for local block sparse attention.
            Default 65536.
        tiling_stgy_name(str): A str to define tiling strategy of flash attention.
        dp(int): data parallel.
            Default 1.
        mp(int): model parallel.
            Default 1.
        have_attention_mask_batch(bool): indicates whether attention_mask contains the batch dimension.
            Default True
        alibi(bool): This parameter indicates whether the flashattention supports the Alibi.
            Default: False


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
        >>> from mindspore import dtype as mstype
        >>> from accspeed.nn.layer.flash_attention import FlashAttention
        >>> from mindspore import Tensor
        >>> model = FlashAttention(head_dim=128,
        ...                        dropout_rate=0.1,
        ...                        prev_block_num=7,
        ...                        next_block_num=0
        ...                        )
        >>> query = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> key = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> value = Tensor(np.ones((2, 16, 4096, 128)), mstype.float16)
        >>> attention_mask = Tensor(np.triu(np.ones((1, 128, 128)), k=1), mstype.float16)
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

* ut测试：进入tests/train/flash_attention/ut目录
* st测试：进入tests/train/flash_attention/st目录

全量测试运行 `pytest -sv .` ; 测试单个文件或函数用例`pytest -sv test_XXX.py::test_func_name`即可