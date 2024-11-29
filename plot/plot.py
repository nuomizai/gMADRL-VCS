import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
def smooth(data, window=250):
    """使用滑动平均对数据进行平滑处理"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main(CONF):
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Performance Metrics')
    
    # 初始化存储所有数据的字典
    all_metrics = {name: {} for name in CONF['name_list']}
    
    # 首先加载所有数据
    for file_path, name in zip(CONF['file_path'], CONF['name_list']):
        data = np.load(file_path)
        all_metrics[name] = {
            'Data Collection': data['dcr'],
            'Fairness': data['fairness'],
            'Energy Consumption Ratio': data['ecr'],
            'Energy Efficiency': data['eff']
        }
    
    # 在子图中绘制每个指标
    for (metric_name, ax) in zip(['Data Collection', 'Fairness', 'Energy Consumption Ratio', 'Energy Efficiency'], axes.flat):
        for name in CONF['name_list']:
            # 获取默认的颜色循环中的下一个颜色
            color = next(ax._get_lines.prop_cycler)['color']
            values = all_metrics[name][metric_name]
            
            # 使用相同颜色绘制原始数据（带透明度）
            ax.plot(values, alpha=0.3, label=f'{name}-Original', color=color)
            
            # 使用相同颜色绘制平滑后的数据（不透明）
            smoothed = smooth(values)
            ax.plot(np.arange(len(smoothed)), smoothed, 
                   linewidth=2, label=f'{name}-Smoothed', color=color)
        
        ax.set_title(metric_name)
        ax.set_xlabel('Steps')
        ax.grid(True)
        ax.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_path = 'comparison_plot.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == '__main__':
    CONF = {
        'file_path': ['../exp_local/2024.11.29/dataset=ROMA,uav_n=4,ugv_n=4/2/save_data.npz'],
        'name_list': ['gMADRL-VCS']
    }
    
    main(CONF)
