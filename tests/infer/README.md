# 推理测试指导
## 环境准备
1、CANN-toolkit：7.0.RC1.B100 <br>
2、环境变量: SOC_VERSION。即NPU芯片型号，目前支持Ascend310P或者Ascend910<br>
```bash
export SOC_VERSION=Ascend310P
或者
export SOC_VERSION=Ascend910
```

3、python依赖包：详见项目根目录[requirements.txt](../../requirements.txt)<br>

注意：安装pytorch后需要侵入式修改pytorch内一处脚本代码，测试可正常运行，可通过`pip show torch`查看安装目录：<br>
修改`torch安装路径/torch/onnx/utils.py` 中找到`_check_onnx_proto(proto)`函数 <br>
```python
_check_onnx_proto(proto)
改为
pass
```

4、ais_bench工具，安装方法详见[链接](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)


## 运行测试
进入项目根目录下的`tests/infer/flash_attention/tik`，执行run_test.sh脚本。
```bash
bash run_test.sh

```
