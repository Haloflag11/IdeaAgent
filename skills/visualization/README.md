# Visualization Skill

可视化机器学习实验结果并生成 Markdown 报告。

## 安装依赖

在使用此技能之前，需要安装必要的 Python 库：

### 方法 1: 使用 requirements.txt
```bash
cd skills/visualization
pip install -r requirements.txt
```

### 方法 2: 直接安装
```bash
pip install matplotlib seaborn pandas numpy
```

## 使用方法

### 基本使用
```bash
# 运行示例（使用默认数据）
python script.py

# 指定输出目录
python script.py -o my_results

# 静默模式（不显示图形界面）
python script.py --quiet
```

### 输出结构

运行后会生成以下文件：
```
visualization_output/
├── images/
│   ├── training_curves.png        # 训练曲线图
│   ├── model_comparison.png       # 模型对比图
│   ├── feature_importance.png     # 特征重要性图
│   └── ...
└── experiment_report.md           # 完整的实验报告
```

## 功能特性

### 1. 训练曲线可视化
- 支持多指标同时显示（loss, accuracy 等）
- 自动图例和网格
- 高分辨率输出（300 DPI）

### 2. 模型对比柱状图
- 自动数值标签
- 美观的配色方案
- 支持任意数量模型对比

### 3. 混淆矩阵热力图
- 自动标注数值
- 颜色映射优化
- 支持多分类任务

### 4. 特征重要性图
- 自动排序（重要性降序）
- 支持显示 Top-N 特征
- 水平条形图更易阅读

### 5. Markdown 报告生成
- 实验概述
- 性能指标表格
- 可视化结果嵌入
- 分析与讨论
- 结论与建议

## 在 IdeaAgent 中使用

当你的实验任务需要可视化结果时，IdeaAgent 会自动使用此技能：

1. 实验完成后，Agent 会调用此技能
2. 生成所有必要的可视化图表
3. 创建包含图表和分析的完整报告
4. 保存到你指定的目录

## 自定义报告

你可以修改 `script.py` 中的 `generate_report()` 函数来自定义报告格式：

```python
# 添加自定义章节
report.append("\n## 5. 未来工作\n")
report.append("...\n")
```

## 故障排除

### 问题：中文字体显示异常
**解决**：确保系统已安装中文字体，或修改脚本中的字体设置：
```python
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
```

### 问题：保存图片时出错
**解决**：检查输出目录是否有写入权限，或尝试更改输出路径。

### 问题：依赖安装失败
**解决**：尝试使用 conda 安装：
```bash
conda install matplotlib seaborn pandas numpy
```

## 示例输出

查看生成的 `experiment_report.md` 文件，你会看到类似以下的报告：

```markdown
# 实验报告

**生成时间**: 2026-03-05 14:30:00

## 1. 实验概述
**实验目的**: 比较不同回归模型在预测任务上的性能
**数据集**: 示例数据集 (1000 样本，15 特征)

## 2. 实验结果
### 2.1 性能指标
| 指标 | 数值 |
|------|------|
| Best Model | Gradient Boosting |
| Best R² Score | 0.94 |

### 2.2 可视化结果
#### Training Curves
![Training Curves](images/training_curves.png)

#### Model Comparison
![Model Comparison](images/model_comparison.png)

## 3. 分析与讨论
...

## 4. 结论
...
```

## 许可证

MIT License
