import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 设置中文字体
font = FontProperties(fname=r'/System/Library/Fonts/PingFang.ttc')
plt.rcParams['font.family'] = ['PingFang HK']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_csv('body_metrics.csv')

# 计算基本统计量
def calculate_statistics(df):
    stats = {}
    for column in df.columns:
        if column != 'id':  # 跳过ID列
            mean = df[column].mean()
            std = df[column].std()
            stats[column] = {
                'mean±std': f'{mean:.2f}±{std:.2f}',
                'range': f'{df[column].min():.2f}-{df[column].max():.2f}',
                'p5-p95': f'{df[column].quantile(0.05):.2f}-{df[column].quantile(0.95):.2f}',
                'p25-p75': f'{df[column].quantile(0.25):.2f}-{df[column].quantile(0.75):.2f}',
                'median': f'{df[column].quantile(0.50):.2f}'
            }
    return pd.DataFrame(stats)

def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.drop('id', axis=1).corr()
    
    # 设置中文标签
    labels = ['身高', '头高', '头宽', '头围', '瞳高', '瞳距', '肩宽']
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                xticklabels=labels,
                yticklabels=labels)
    
    plt.title('人体测量指标相关性矩阵', fontproperties=font, fontsize=14)
    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_scatter_matrix(df_clustered, feature_names):
    """绘制散点图矩阵，不同聚类用不同颜色表示"""
    # 设置中文标签
    df_plot = df_clustered.drop('id', axis=1)
    df_plot.columns = ['身高', '头高', '头宽', '头围', '瞳高', '瞳距', '肩宽', 'cluster']
    
    # 设置颜色映射
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    size_labels = ['S码', 'M码', 'L码']
    palette = {i: colors[i] for i in range(3)}
    
    # 创建散点图矩阵
    plt.figure(figsize=(15, 15))
    scatter_plot = sns.pairplot(
        df_plot,
        hue='cluster',
        palette=palette,
        hue_order=[0, 1, 2],
        plot_kws={'alpha': 0.6},
        diag_kind='hist'
    )
    
    # 设置图例标签
    new_labels = size_labels
    legend = scatter_plot._legend
    if legend is not None:
        for t, l in zip(legend.texts, new_labels):
            t.set_text(l)
    
    # 设置标题和标签字体
    for ax in scatter_plot.axes.flat:
        ax.set_xlabel(ax.get_xlabel(), fontproperties=font)
        ax.set_ylabel(ax.get_ylabel(), fontproperties=font)
    
    plt.savefig('scatter_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_size_ranges(df):
    """计算各个尺码的区间"""
    size_metrics = {}
    
    # 定义三个尺码
    sizes = ['S', 'M', 'L']
    
    # 使用33.33%和66.67%作为分界点
    percentiles = [0, 33.33, 66.67, 100]
    
    for column in df.columns:
        if column != 'id':
            bounds = np.percentile(df[column], percentiles)
            size_ranges = {}
            
            for i, size in enumerate(sizes):
                size_ranges[size] = {
                    'range': f'{bounds[i]:.1f}-{bounds[i+1]:.1f}',
                    'mean': f'{df[column][(df[column] >= bounds[i]) & (df[column] < bounds[i+1])].mean():.1f}'
                }
            
            size_metrics[column] = size_ranges
    
    return size_metrics

def print_size_table(size_metrics):
    """打印尺码表"""
    for metric, sizes in size_metrics.items():
        print(f"\n{metric} 尺码表:")
        print("-" * 60)
        print(f"尺码    {'范围':^15}    {'平均值':^10}")
        print("-" * 60)
        for size, data in sizes.items():
            print(f"{size:^6} {data['range']:^15} {data['mean']:^10}")

def save_size_ranges(size_metrics, filename='size_ranges.csv'):
    """保存尺码表到CSV文件"""
    rows = []
    for metric, sizes in size_metrics.items():
        for size, data in sizes.items():
            rows.append({
                '指标': metric,
                '尺码': size,
                '范围': data['range'],
                '平均值': data['mean']
            })
    
    df_sizes = pd.DataFrame(rows)
    df_sizes.to_csv(filename, encoding='utf-8-sig', index=False)

def plot_size_distributions(df):
    """绘制尺码分布图"""
    plt.figure(figsize=(15, 10))
    
    metrics = df.columns[1:]  # 跳过id列
    n_metrics = len(metrics)
    rows = (n_metrics + 1) // 2
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(rows, 2, i)
        sns.histplot(data=df, x=metric, bins=30)
        plt.title(f'{metric}分布', fontproperties=font)
        plt.xlabel('数值', fontproperties=font)
        plt.ylabel('频数', fontproperties=font)
    
    plt.tight_layout()
    plt.savefig('size_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def perform_kmeans_analysis(df, n_clusters=3):
    """使用K-means对用户进行聚类分析"""
    # 准备数据
    features = df.drop('id', axis=1)
    feature_names = features.columns
    
    # 标准化数据 - 使用RobustScaler来处理异常值
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 设置特征权重 - 给身高和肩宽更高的权重
    weights = np.ones(len(feature_names))
    weights[features.columns.get_loc('身高')] = 2.0  # 身高权重
    weights[features.columns.get_loc('肩宽')] = 1.5  # 肩宽权重
    
    # 应用权重
    weighted_features = scaled_features * weights
    
    # 执行K-means聚类，增加迭代次数和初始化次数
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20,        # 增加初始化次数
        max_iter=500,     # 增加最大迭代次数
        init='k-means++'  # 使用k-means++初始化方法
    )
    cluster_labels = kmeans.fit_predict(weighted_features)
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(weighted_features, cluster_labels)
    
    # 将原始中心点转换回原始尺度
    centers = scaler.inverse_transform(kmeans.cluster_centers_ / weights)
    
    # 添加聚类标签到原始数据
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    
    # 根据身高对聚类进行排序，确保cluster 0是最矮的，cluster 2是最高的
    cluster_heights = df_clustered.groupby('cluster')['身高'].mean()
    cluster_order = cluster_heights.argsort()
    cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
    df_clustered['cluster'] = df_clustered['cluster'].map(cluster_mapping)
    
    # 重新排序中心点
    centers = centers[cluster_order]
    
    return df_clustered, centers, silhouette_avg, feature_names

def analyze_clusters(df_clustered, cluster_centers, feature_names):
    """分析每个聚类的特征"""
    cluster_stats = []
    size_labels = ['S', 'M', 'L']  # 添加尺码标签
    
    for cluster in range(len(cluster_centers)):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster]
        
        # 计算该聚类的统计信息
        stats = {
            'size': len(cluster_data),
            'percentage': f"{(len(cluster_data) / len(df_clustered) * 100):.1f}%",
            'size_label': size_labels[cluster],  # 添加尺码标签
            'features': {}
        }
        
        # 计算每个特征的统计量
        for i, feature in enumerate(feature_names):
            stats['features'][feature] = {
                'center': cluster_centers[cluster][i],
                'mean': cluster_data[feature].mean(),
                'std': cluster_data[feature].std(),
                'min': cluster_data[feature].min(),
                'max': cluster_data[feature].max()
            }
        
        cluster_stats.append(stats)
    
    return cluster_stats

def plot_cluster_characteristics(df_clustered, feature_names):
    """绘制聚类特征分布图"""
    n_features = len(feature_names)
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    size_labels = ['S', 'M', 'L']
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    
    for i, feature in enumerate(feature_names, 1):
        plt.subplot(n_rows, n_cols, i)
        for cluster in sorted(df_clustered['cluster'].unique()):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster]
            sns.kdeplot(
                data=cluster_data[feature], 
                label=f'{size_labels[cluster]}码',
                color=colors[cluster],
                warn_singular=False
            )
        
        plt.title(f'{feature}的尺码分布', fontproperties=font)
        plt.xlabel('数值', fontproperties=font)
        plt.ylabel('密度', fontproperties=font)
        plt.legend(prop=font)
    
    plt.tight_layout()
    plt.savefig('cluster_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_cluster_analysis(cluster_stats, filename='cluster_analysis.csv'):
    """保存聚类分析结果到CSV"""
    rows = []
    for i, stats in enumerate(cluster_stats):
        for feature, values in stats['features'].items():
            rows.append({
                '聚类组': f'聚类 {i}',
                '样本数量': stats['size'],
                '占比': stats['percentage'],
                '指标': feature,
                '中心值': f"{values['center']:.2f}",
                '平均值': f"{values['mean']:.2f}",
                '标准差': f"{values['std']:.2f}",
                '范围': f"{values['min']:.1f}-{values['max']:.1f}"
            })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(filename, encoding='utf-8-sig', index=False)

if __name__ == "__main__":
    # 计算并显示统计量
    stats = calculate_statistics(df)
    print("\n基本统计量:")
    print(stats.T)
    
    # 保存统计结果到CSV
    stats.T.to_csv('metrics_statistics.csv', encoding='utf-8-sig')
    
    # 生成相关性矩阵图
    plot_correlation_matrix(df)
    
    # 计算并显示相关性系数
    correlation = df.drop('id', axis=1).corr()
    print("\n相关性系数矩阵:")
    print(correlation.round(3))
    correlation.round(3).to_csv('correlation_matrix.csv', encoding='utf-8-sig')
    
    print("\n执行K-means聚类分析...")
    df_clustered, centers, silhouette_avg, feature_names = perform_kmeans_analysis(df)
    print(f"轮廓系数: {silhouette_avg:.3f}")
    
    # 移动散点图矩阵的绘制到聚类之后
    plot_scatter_matrix(df_clustered, feature_names)
    
    # 分析聚类结果
    cluster_stats = analyze_clusters(df_clustered, centers, feature_names)
    
    # 打印聚类结果
    print("\n聚类分析结果:")
    for i, stats in enumerate(cluster_stats):
        print(f"\n聚类 {i} (样本数: {stats['size']}, 占比: {stats['percentage']})")
        print("-" * 60)
        for feature, values in stats['features'].items():
            print(f"{feature:>10}: {values['mean']:.1f}±{values['std']:.1f} "
                  f"({values['min']:.1f}-{values['max']:.1f})")
    
    # 保存聚类结果
    save_cluster_analysis(cluster_stats)
    
    # 绘制聚类分布图
    plot_cluster_characteristics(df_clustered, feature_names) 