# 推理编译部署指导

## 环境准备
1、CANN-toolkit：7.0.RC1.B100 <br>
2、环境变量: SOC_VERSION即NPU芯片型号，目前支持Ascend310P或者Ascend910
```bash
export SOC_VERSION=Ascend310P
或者
export SOC_VERSION=Ascend910
```

## 编译部署

### 执行编译脚本
进入项目根目录下的infer/build_infer目录，执行build_ops.sh脚本
```bash
bash build_ops.sh
```
该脚本会使用当前实际执行环境的 msopgen 工具在目录下生成 custom_project 自定义算子工程，然后将tik交付件拷贝到算子自定义工程进行构建和打包。
打包完成后在当前目录生成**aie_ops.run**文件，即为构建好的run包。


### 安装
默认在打包完成后自动部署至当前目录下的pkg/vendors/customize,需要执行set_env.sh脚本设置环境变量后即可使用。
默认情况下直接执行以下命令即可：
```bash
source set_env.sh
```
如需要安装至其他自定义目录，则执行以下命令：
```bash
dst=/path/to/install  # 请修改'/path/to/install'为想要安装的路径
bash aie_ops.run --extract=${dst}
export ASCEND_CUSTOM_OPP_PATH=${dst}/vendors/customize:$ASCEND_CUSTOM_OPP_PATH
```


