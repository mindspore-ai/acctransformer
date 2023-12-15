# MindSpeed

## FlashAttention

### 环境
#### 一、配套环境一
1. cann version：version=6.0.RC1
2. mindspore version：1.9.0
#### 二、配套环境二
1. cann version：version=6.3.RC2
2. mindspore version：r2.1

### FlashAttention使用方法

* 环境安装：
1. for cann version：version=6.0.RC1
```bash
source tik_v200_env.sh`
```
2. for cann version：version=6.3.RC2
```bash
export PYTHONPATH=/home/zcs/codes/Mindspeed/accspeed:$PYTHONPATH
```

* 单算子调试：进入目录tbe/flash_attention，运行 `python flash_attention.py`
* ut测试：进入tbe/ut目录，运行 `pytest -v test_flash_attention.py` ; 保证测试数据完整；
* st测试：进入tbe/st目录，运行 `pytest -v -s test_flash_attention.py` ; 测试单个函数 `test_flash_attention.py`::函数名，即可；

### Wukong-Huahua模型训练推理

* 进入model_wukong-huahua目录
* 准备模型文件：models/wukong-huahua-ms.ckpt; 参考：`/home/ranjiewen/mind_speed/model_wukong-huahua/models`
* 准备微调数据集：dataset/ ； `参考： /home/ranjiewen/mind_speed/model_wukong-huahua/dataset`
* 推理脚本：`sh scripts/infer.sh`
* 单卡训练脚本：`sh scripts/run_train.sh`
