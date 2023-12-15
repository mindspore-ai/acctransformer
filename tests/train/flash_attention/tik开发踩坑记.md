### TIK & 达芬奇架构

* TIK（Tensor Iterator Kernel）是一种基于Python语言的动态编程框架，呈现为一个Python模块。开发者可以通过调用TIK提供的API基于Python语言编写自定义算子，然后TIK编译器会编译为适配昇腾AI处理器或BS9SX1A AI处理器SoC应用程序的二进制文件。

### 开发踩坑记

* 汇总记录开发过程中的问题，方便查询


### 精度问题

1. ut测试：

> 1.1 h_reduce_sum 使用fp32；
>
> 1.2 vec_rec 倒数使用高精度接口vec_rec_high_preci， h_div；

2. softmax: 归一化后精度不够  **Pij / li** ：

> 2.1 ms 实现softmax 与 ms.ops.softmax 精度对齐， 进行类型对齐，tik实现1e-5精度左右；
>
> 2.2 调整更新O顺序：（1）先做系数乘ac + bc，（2）先做括号内部( a + b) * c，在Pij_Vj传进去情况下可能精度影响不大；
>
> 2.3 Pij / li问题，softmax内部cast fp16, 外面fp32影响精度1e-3左右；
>
> 2.4 st尝试不分块, softmax做除法, 然后进行Pij_ub * Vj, 用例1e-3过不了；
>
> 2.5  (scale * Pij) * Vj 与 scale * (Pij * Vj) tik fp16实现精度1e-3左右；

3. 总结&todo

> 3.1 Pij, l, m，vector都使用fp32,（h_reduce_max不支持fp32）,成本代价高；
>
> 3.2 查看GPU版推理算子是否有精度问题
>
> 3.3 Pij / li 除法位置，(scale * Pij) * Vj 乘法顺序，分块累计误差
>
