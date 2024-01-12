# FA与TA对比用例执行

## 单卡单算子用例运行
注意：FA与TA对比测试使用整网训练dump的q,k,v作为测试输入，执行测试前需要修改dump的q,k,v本地文件路径

**test_fa_grad_precision.py**
```python

@pytest.mark.parametrize(
    "q_npy_path, k_npy_path, v_npy_path",
    [
        ("savepath/q.npy", "savepath/k.npy", "savepath/v.npy"),
    ],
)
def test_fa_grad_compare_with_triangle_attn_grad(q_npy_path, k_npy_path, v_npy_path):

```

单卡单算子用例执行命令如下：
```bash

pytest -sv test_fa_grad_precision.py::test_fa_grad_compare_with_triangle_attn_grad

```

## 多卡单算子用例运行
注意：FA与TA对比测试使用整网训练dump的q,k,v作为测试输入，执行测试前需要修改dump的q,k,v本地文件路径；多卡并行测试除了修改q,k,v文件路径，还需要修改ranktable文件路径以及dp,mp分布式配置参数

1. **test_fa_grad_precision_distribute.py**
```python

@pytest.mark.parametrize(
    "q_npy_path, k_npy_path, v_npy_path, dp, mp",
    [
        ("savepath/q.npy", "savepath/k.npy", "savepath/v.npy", 1, 8), # 按实际情况修改
    ],
)
def test_fa_grad_compare_with_triangle_attn_grad(q_npy_path, k_npy_path, v_npy_path, dp, mp):


```

2. **run_fa_grad_precision_distribute.bash**，修改RANK_TABLE_FILE环境变量为实际的ranktable文件路径
```bash

export RANK_TABLE_FILE=${EXEC_PATH}/ranktables/ranktable_8p.json

```

多卡单算子用例执行命令如下：
```bash

bash run_fa_grad_precision_distribute.bash

```
执行后会打印日志路径，查看日志即可获取测试结果