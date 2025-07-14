import numpy as np
# i = (1, 0)
# j = (1, 2)
# k = (3, 3)

# x_space = np.linspace(0, 1, 5 + 1, endpoint=True)
# y_space = np.linspace(0, 1, 5 + 1, endpoint=True)
# yy, xx = np.meshgrid(y_space, x_space)
# #先y 后x 是方便将网格按照坐标定位的 比如 i(1, 0) 指的是x=1 y=0 的位置即点[0.2, 0] 而非第一行第0 列的意思
# xy = np.stack([xx, yy], axis=2)
# # xx ,yy = np.meshgrid(x_space, y_space)

# def get(i):
#     return xy[i[0], i[1]] #上面那样的网格设置也就可以保证 i[0]指的是 x 坐标，i[1]指的是 y 坐标，

# i = get(i)
# j = get(j)
# k = get(k)
# print('i', i)
# print('j', j)
# print('k', k)
# print('j - k', j - k)
# print('(i - k) * (k - j)', (i - k) * (k - j))
# print(np.dot((i - k),(k - j)))

# def dot_product(x, y):
#     return np.dot(x, y)
# array = np.array( [ [(j - k) ** 2, (i - k) * (k - j), (i - j) * (j - k)],
#              [(i - k) * (k - j), (i - k) ** 2, (k - i) * (i - j)],
#              [(i - j) * (j - k), (k - i) * (i - j), (i - j) ** 2]  ])

# sum = np.sum(array, axis=2)

# sum_2 = np.array(
#     [ [dot_product(j-k, j-k), dot_product(i-k, k-j), dot_product(i-j, j-k)],
#       [dot_product(i - k, k - j), dot_product(i - k, i - k), dot_product(k - i, i - j)],
#       [dot_product(i - j, j - k), dot_product(k - i, i - j), dot_product(i - j, i - j)]
#       ] )

# array_2 = np.array([[(j-k, j-k), (i-k, k-j), (i-j, j-k), (i-k,j-k)],
#            [(j-k, j-k), (i-k, k-j), (i-j, j-k), (i-k,j-k)]])

# array_new = np.zeros(array_2.shape)

# # for i, row in range(1, array_2.shape[0]+1):
# #     for j in range(1, array_2.shape[1]+1):
# #         array_new[i][j] = [dot_product(array_2[i][j][0], array_2[i][j][1])]
# # #
# # print('array', array)
# # print('sum', sum)
# # print('sum_2', sum_2)




# # 全局刚度矩阵
# global_stiffness = np.zeros((3, 3))

# # 局部刚度矩阵
# local_stiff = np.array([[6, 8],
#                         [7, 9]])

# # 指定要添加局部刚度矩阵的索引位置
# sub_idx = np.array([[0, 0],
#                     [2, 1]])

# # 使用 np.ix_ 创建索引器
# idx = np.ix_(sub_idx[:, 0], sub_idx[:, 1])

# # 将局部刚度矩阵加到全局刚度矩阵的指定位置
# global_stiffness[idx] += local_stiff

# print('global_stiffness', global_stiffness)


