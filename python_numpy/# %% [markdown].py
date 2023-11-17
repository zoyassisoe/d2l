# %% [markdown]
# # 2.1与2.2

# %%
import numpy as np

# %%
# 创建
a = np.array([range(10)], dtype=np.float32)
b = np.zeros(10, dtype=np.int32)
c = np.ones((3,3))
d = np.full((3,3), 3.14)
a, b, c, d

# %%
x1 = np.arange(0, 20, 2)
x2 = np.linspace(0, 1, 5)
x1, x2

# %%
# 0-1均匀分布随机生成
q1 = np.random.random((3, 3))
# Parameters
# ----------
# loc : float or array_like of floats
#     Mean ("centre") of the distribution.
# scale : float or array_like of floats
#     Standard deviation (spread or "width") of the distribution.
# size
q2 = np.random.normal(0, 1, (3, 3))
q3 = np.random.randint(10, size=(5, 5))
# 生成单位阵
q4 = np.eye(3)
q1, q2, q3

# %%
# 元素访问
# x[start:stop:step]
x = np.arange(1, 10, 1)
y = np.arange(1, 17, 1).reshape(2, 2, 4)
(x, x[::2], x[:5:]), (y, y[0,1,1], y[:,:,::2], )

# %%
# 注意视图与副本
# numpy的数组切片保存的是视图
x = np.arange(1, 10, 1)
y1 = x[:2]
y2 = x[:2].copy()
x[0] = 10
x, y1, y2

# %%
# reshape
x = np.array([1,2,3,4]).reshape((-1, 2, 2))
x.shape, x
# np.newaxis???

# %%
# 数组拼接与分裂
# 向量连接与张量连接有区别
x = np.arange(3).reshape(-1, 3)
y = np.arange(3).reshape(-1, 3)
xy = np.concatenate([x,y], axis=0)
x, y, xy, np.split(xy, [1,2], axis=1)


