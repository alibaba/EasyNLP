from pylab import *
# mpl.rcParams['font.sans-serif'] = ['STIXGeneralBolIta']  # 添加这条可以让图形显示中文

# x_axis_data = ["0.5k", "1k", "2k", "5k", "","10k"]
# y1_axis_data = [44.76, 47.31, 48.12, 49.73, 51.46, 51.10]
# y2_axis_data = [48.31, 51.58, 55.61, 59.53, 61.12, 62.06]

x_axis_data = [1, 2, 4, 8, 16]
y1_axis_data = [20.46, 25.38, 26.67, 27.72, 28.14]
y2_axis_data = [23.31, 28.35, 33.99, 42.66, 48.30]

# plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
plt.plot(x_axis_data, y1_axis_data, 'ro-', color='blue', alpha=0.8, linewidth=1, label='Full FT')
plt.plot(x_axis_data, y2_axis_data, 'bs-', color='red', alpha=0.8, linewidth=1, label='XtremeCLIP')

plt.xscale('log', base=2)
from matplotlib.ticker import ScalarFormatter

ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter())
# 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
plt.legend(loc="upper right")
# plt.xlabel('#training samples', fontsize = 18)
plt.xlabel('#shot', fontsize = 18)
plt.ylabel('accuracy(%)', fontsize = 18)
font1 = {'weight': 'normal','size': 16}
plt.legend(loc="lower right", prop=font1)
plt.title("Image Classification (FGVC)", fontsize = 25)
# plt.title("Visual Entailment", fontsize = 20)
plt.tick_params(labelsize=13)
plt.xticks(x_axis_data)
# plt.xticks(x_axis_data)
# plt.show()
# plt.savefig("/home/moming/ve_V4.png", dpi=300)
plt.savefig("/home/moming/fgvc.png", dpi=300)
# plt.savefig('demo.jpg')  # 保存该图片
