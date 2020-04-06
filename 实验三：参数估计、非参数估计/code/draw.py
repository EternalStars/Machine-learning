import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

price = [93.33, 96.00, 96.00, 95.33, 96.67, 96.00, 97.33]
"""
绘制水平条形图方法barh
参数一：y轴
参数二：x轴
"""
plt.barh(range(7), price, height=0.7, color='steelblue', alpha=0.8)      # 从下往上画
plt.yticks(range(7), ['最近邻决策', '核函数gaussian', '核函数tophat', '核函数epanechnikov', '核函数exponential', '核函数linear', '参数估计'])
plt.xlim(90, 100)
plt.xlabel("正确率")
plt.title("不同估计方法正确率对比条形图")
for x, y in enumerate(price):
    plt.text(y + 0.2, x - 0.1, '%s' % y)
plt.show()
