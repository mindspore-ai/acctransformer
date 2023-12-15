# 用例执行
## 1.1 全量执行所有用例
切换到st/ut目录下，执行以下命令：
```bash
pytest -sv . 
```
## 1.2 执行非耗时用例
```bash
# -s 表示打印用例中输出信息.
# -m 根据用例 mark 筛选
pytest -sv . -m 'not time_consuming'
```

## 1.3 执行算子逻辑无关的非耗时用例
```bash
pytest -v -m 'not time_consuming and not experiment' .
```

## 1.4 统计所有用例耗时情况
```bash
pytest -v -m 'not time_consuming' --durations=0
```