# income-data-analyse

UCI Adult (Census Income) 数据集的完整 **EDA + 机器学习建模**。数据为 `raw-data/adult.data`（无表头 CSV，15 列，32 561 条记录）。

## 环境

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python \
    pandas numpy matplotlib seaborn scikit-learn jupyter nbconvert ipykernel
.venv/bin/python -m ipykernel install --user --name income-data-analyse \
    --display-name "Python (income-data-analyse)"
```

## 运行

```bash
.venv/bin/python build_notebook.py
```

此脚本会：

1. 拼接 `income-analysis.ipynb`（31 个 cell，含中文 Markdown 解释）
2. 用注册的 `income-data-analyse` kernel 顺序执行全部 cell
3. 导出 `income-analysis.html` 中文报告

## 产物

| 路径 | 说明 |
| --- | --- |
| `income-analysis.ipynb` | 完整 Notebook（31 cells） |
| `income-analysis.html` | 导出的中文报告（~1.9 MB，含全部图表） |
| `figures/01_*.png` … `13_*.png` | 13 张分析图表 |
| `build_notebook.py` | 组装 / 执行 / 导出脚本 |

## 分析流程

1. **数据载入与清洗**：列名加注、识别 `?` 为 NaN、去重
2. **单变量 EDA**：数值列描述统计 / 直方图 / 箱线图；类别列 top-N 频次
3. **双变量 EDA**：数值特征按 income 分布对比；类别特征 vs `>50K` 占比
4. **多变量 EDA**：数值相关热力图；`education × marital-status → >50K` 交叉热力图
5. **预处理**：类别众数填补 + One-Hot 编码 + 80/20 分层划分
6. **建模**：Logistic / DecisionTree / RandomForest / GradientBoosting 的 5 折 CV 与测试集评估
7. **模型解释**：Random Forest 特征重要性 + Permutation Importance
8. **结论**：含伦理提示（性别 / 种族差异与公平性）

## 关键结果

| 模型 | CV AUC | Test AUC | Test F1 | Test Acc |
| --- | --- | --- | --- | --- |
| GradientBoosting | 0.9200 | **0.9272** | **0.7015** | **0.8719** |
| RandomForest | 0.9026 | 0.9113 | 0.6955 | 0.8628 |
| DecisionTree | 0.8923 | 0.9031 | 0.6764 | 0.8634 |
| LogisticRegression | 0.8881 | 0.8966 | 0.6568 | 0.8502 |

最重要的预测因子：`capital-gain` / `capital-loss` > `age` / `hours-per-week` > `marital-status` / `relationship` > `education-num` 及高学历 One-Hot。
