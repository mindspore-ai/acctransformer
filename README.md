# acctransformer 介绍

## 一、介绍
**acctransformer**是一个基于MindSpore框架以及昇腾 CANN 软件栈的transformer加速库，原生支持昇腾AI处理器NPU。<br>
实现了一些对transformer模型中self-attention部分的加速算法，目前已支持:
* **FlashAttention2**

如果您对MindSpore acctransformer有任何建议，请通过issue与我们联系，我们将及时处理。

算法支持列表如下：

| 名称 | 路径 | 文档 |
| --- | --- | --- |
| FlashAttention2 | [FlashAttention2](train/flash_attention) | [文档](train/flash_attention/README.md) |

## 二、安装使用
### 2.1、环境安装
#### 2.1.1、配套环境要求
首先需要准备包含昇腾AI处理器NPU的Linux服务器，并安装对应CANN版本的NPU驱动以及固件。

算法配套环境表如下：

| 名称 | 配套组件 | 版本要求 |
| --- | --- | --- |
| FlashAttention2 | MindSpore<br>CANN配套软件包<br>NPU: Ascend 910 | MindSpore: [2.2.0](https://www.mindspore.cn/versions#2.2.0) <br> CANN配套软件包: 适配MindSpore版本|

#### 2.1.2、安装指南

MindSpore官方网站：[链接](https://www.mindspore.cn/install) <br>
各算法安装以及使用方法，参考算法支持列表各目录下README文档。

## 三、分支以及版本说明
初始版本，后续待补充

## 四、测试

参考每个算法模块下README指导文档。

## 五、许可证
[Apache License 2.0](LICENSE)