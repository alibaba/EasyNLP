import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 这里是创建一个数据
pic_content = ["Highway", "Industrial", "Pasture", "Residential", "River"]
cate_name = ["Highway", "Industrial", "Pasture", "Residential", "River"]

# prob_dist = np.array([[0.1002, 0.0805, 0.0905, 0.1296, 0.1113, 0.1050, 0.1030, 0.1010, 0.1109, 0.0678],
#                       [0.0996, 0.0770, 0.0886, 0.1133, 0.1294, 0.1039, 0.1036, 0.1102, 0.1077, 0.0668],
#                       [0.0992, 0.0712, 0.0965, 0.1111, 0.1064, 0.1259, 0.1169, 0.1064, 0.1060, 0.0604],
#                       [0.0964, 0.0790, 0.1019, 0.1011, 0.1093, 0.0991, 0.1068, 0.1318, 0.1023, 0.0723],
#                       [0.1014, 0.0942, 0.0927, 0.1084, 0.1018, 0.1059, 0.0956, 0.0979, 0.1190, 0.0831],
#                       ])
prob_dist = np.array([[0.1296, 0.1113, 0.1050, 0.1010, 0.1109],
                      [0.1133, 0.1294, 0.1039, 0.1102, 0.1077],
                      [0.1111, 0.1064, 0.1259, 0.1064, 0.1060],
                      [0.1011, 0.1093, 0.0991, 0.1318, 0.1023],
                      [0.1084, 0.1018, 0.1059, 0.0979, 0.1190],
                      ])
# prob_dist = np.array([[8.3447e-07, 0.0000e+00, 5.9605e-08, 9.9951e-01, 1.7226e-05, 5.9605e-08, 2.1279e-05, 2.4438e-06, 5.7459e-04, 0.0000e+00],
#                       [5.6028e-06, 5.9605e-08, 5.9605e-08, 1.4079e-04, 9.9902e-01, 1.7107e-05, 7.5936e-05, 2.9802e-04, 6.3562e-04, 1.1921e-07],
#                       [1.0729e-06, 1.7881e-07, 1.1921e-07, 7.2122e-06, 1.0133e-06, 1.0000e+00, 3.9458e-05, 2.6286e-05, 6.0201e-06, 5.9605e-08],
#                       [2.3842e-07, 0.0000e+00, 2.2650e-06, 4.0531e-06, 3.1173e-05, 1.7881e-07, 5.1546e-04, 9.9951e-01, 4.7684e-07, 4.1723e-07],
#                       [3.6359e-06, 1.7881e-07, 2.9802e-07, 9.4593e-05, 4.2319e-06, 6.5565e-07, 2.3842e-07, 6.9141e-06, 1.0000e+00, 2.9206e-06],
#                       ])
# prob_dist = np.array([[9.9951e-01, 1.7226e-05, 5.9605e-08, 2.4438e-06, 5.7459e-04],
#                       [1.4079e-04, 9.9902e-01, 1.7107e-05, 2.9802e-04, 6.3562e-04],
#                       [7.2122e-06, 1.0133e-06, 1.0000e+00, 2.6286e-05, 6.0201e-06],
#                       [4.0531e-06, 3.1173e-05, 1.7881e-07, 9.9951e-01, 4.7684e-07],
#                       [9.4593e-05, 4.2319e-06, 6.5565e-07, 6.9141e-06, 1.0000e+00],
#                       ])

# 这里是创建一个画布
fig, ax = plt.subplots(figsize=(9,7))
im = ax.imshow(prob_dist)

# 这里是修改标签
# We want to show all ticks...
ax.set_xticks(np.arange(len(cate_name)))
ax.set_yticks(np.arange(len(pic_content)))
# ... and label them with the respective list entries
ax.set_xticklabels(cate_name)
ax.set_yticklabels(pic_content)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# 因为x轴的标签太长了，需要旋转一下，更加好看
# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# 添加每个热力块的具体数值
# Loop over data dimensions and create text annotations.
# for i in range(len(pic_content)):
#     for j in range(len(cate_name)):
#         text = ax.text(j, i, prob_dist[i, j],
#                        ha="center", va="center", color="w")
# ax.set_title("prior knowledge")
plt.title("before fine-tuning",fontsize = 20)
# plt.title("after fine-tuning",fontsize = 20)
fig.tight_layout()
cb=plt.colorbar(im)
cb.ax.tick_params(labelsize=16)
# plt.show()
plt.savefig("/home/moming/heat_prior_v2.png", dpi=300)
# plt.savefig("/home/moming/finetune_V3.png", dpi=300)