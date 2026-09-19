# Tests 1.1–7.6（Python）

共 25 个测试。计算方法参考课程的 `library/` 和 `testfiles/test_setup.jl`。

## 输入和预期输出在哪里？

都在 `testfiles/data/`。`test1.csv` 等是输入，`testout_1.1.csv` 等是预期输出。
每题的对应关系见 `testfiles/Tests.xlsx` 的 Input、Output 两列。
Test 3、4 按题目要求，把前面测试的标准输出作为输入。

本目录的 `outputs/` 是 Python 自己算出来的结果，文件名与标准答案一致。
`outputs/comparison.csv` 记录每题的比对状态和最大误差。

## 运行

在课程仓库根目录运行：

```bash
python3 -m pip install -r testfiles/python/requirements.txt
python3 testfiles/python/run_tests.py
```

`risk_functions.py` 放计算函数，`run_tests.py` 按题号调用它们、保存结果、比对答案。
如果某题不通过，程序会显示 FAIL，并返回非零退出码。

本次验证环境：Python 3.13、NumPy 2.3.3、pandas 2.3.3、SciPy 1.16.2。
25 项通过下面定义的本地检查：20 项数值比对、5 项模拟统计比对。
重复运行后，25 个输出文件和比对汇总的内容完全一致。

## 各题做什么？

- 1.1–1.4：删除缺失行或按变量对处理缺失值，计算协方差、相关系数。
- 2.1–2.3：指数加权协方差、相关系数，以及两种衰减参数的组合。
- 3.1–3.4：用 near-PSD 和 Higham 方法修复矩阵。
- 4.1：对半正定矩阵做 Cholesky 分解。
- 5.1–5.5：每题生成 100,000 个正态样本，比较样本协方差；5.5 使用保留 99% 方差的 PCA。
- 6.1–6.2：计算简单收益率和对数收益率。
- 7.1–7.4：拟合正态分布、t 分布、t 回归，并计算 AICc。
- 7.5–7.6：用矩估计和最大似然分别拟合 NIG 分布。

## 比对时要注意

普通数值比对使用 `绝对误差 <= 1e-8 + 1e-6 × |标准答案|`，列名、维度和日期必须一致。
这是本地检查标准，老师没有在这些文件里提供统一的评分容差。

5.1–5.5 是随机模拟。Python 与 Julia 即使用同一 seed，也不会生成相同样本。
这里固定 Python seed=1234，保证重复运行一致，并采用 5 个标准误的统计检查：
Python 和标准答案各自与目标协方差比对，再比对两者的差异。两份独立模拟之差的
标准误是单份的 √2 倍。因此这些题的 PASS 表示统计上吻合，不是逐位完全相等。
协方差元素的标准误为 `sqrt((C[i,i]*C[j,j] + C[i,j]**2)/(100000-1))`。
修复矩阵的题使用修复后的目标，PCA 题使用保留因子对应的目标。

课程文件有几个容易混淆的地方：

- 2.3 使用方差 λ=0.97、相关系数 λ=0.94，与 Excel、实际 Julia 计算和标准输出一致。
  `test_setup.jl` 中这一题的注释把两个 λ 写反了。
- Excel 部分 Output 文件名有笔误，例如 `testout_6.1.csv` 实际叫 `testout6_1.csv`。
  代码使用 `data/` 内实际存在的名称。
- 7.1 的 sigma 是样本标准差（分母 n−1）。7.2、7.3 的 sigma 是 t 分布的 scale。
- 7.3 的误差位置 mu 固定为 0，Alpha 是回归截距。
- 7.5 使用样本方差（n−1），偏度和超额峰度按课程的未修正中心矩口径计算。
- 7.6 把 SciPy 的 `(a, b, loc, scale)` 换成 `(mu, alpha, beta, delta)`：
  `mu=loc`，`delta=scale`，`alpha=a/scale`，`beta=b/scale`。
  优化结果的末位数字可能随 SciPy 版本变化。
