import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 构造网格
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# 原始曲面 z = x * y
Z1 = X * Y

# 平面 Z_U_1 = 5*X - 5*Y + 25
Z2 = 5 * X - 5 * Y + 25

# 创建图像
fig = plt.figure(figsize=(10, 8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 曲面：z = xy
ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.9, edgecolor='none', label='z = x*y')

# 平面：Z_U_1
ax.plot_surface(X, Y, Z2, color='orange', alpha=0.5, edgecolor='none', label='Z_U_1')

# 坐标轴和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('z = x*y with Overlaid Plane Z_U_1 = 5X - 5Y + 25')

# 提高视角可读性
ax.view_init(elev=30, azim=45)  # 可调角度以获得最佳透视

plt.tight_layout()
plt.show()