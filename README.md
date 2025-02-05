# 人体测量数据分析项目

本项目对人体测量数据进行统计分析和尺码划分，包括身高、头部维度、眼部特征等多个维度的分析。

## 功能特点

- 基础统计分析（均值、标准差、分位数等）
- 相关性分析
- K-means聚类分析
- 可视化分析（散点图、相关性矩阵等）
- 尺码自动划分

## 文件结构

- `analyze_metrics.py`: 主要分析脚本
- `analysis_report.md`: 分析报告
- `body_metrics.csv`: 原始数据
- 输出文件：
  - `metrics_statistics.csv`: 统计分析结果
  - `correlation_matrix.csv`: 相关性分析结果
  - `size_ranges.csv`: 尺码划分结果
  - `cluster_analysis.csv`: 聚类分析结果
  - 可视化图表：
    - `correlation_matrix.png`
    - `scatter_matrix.png`
    - `size_distributions.png`
    - `cluster_distributions.png`

## 使用方法

1. 安装依赖： 