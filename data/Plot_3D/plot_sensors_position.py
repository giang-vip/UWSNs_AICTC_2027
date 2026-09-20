import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ax = plt.figure().add_subplot(projection='3d')
file_path = '../Uniform_Distribution/150_sensors/150_4.txt'
f = open(file_path, 'r')



def plot_sensors_positions():
    depth_sensor = []
    width_sensor = []
    length_sensor = []
    n = int(f.readline())
    for i in range(0, n):
        s = f.readline()
        s = s.split(' ')
        depth_sensor.append(-(float)(s[2]))
        length_sensor.append((float)(s[1]))
        width_sensor.append((float)(s[0]))
    ax.scatter(length_sensor, width_sensor,
               depth_sensor, c='blue', label='sensors')


rmin = 96
w = 2000
d = 1500
dep = [0]
wid = [0]


def plot_square_positions(barrier_l):
    dep = [0]
    wid = [0]
    while dep[len(dep)-1] * rmin < d:
        dep.append(dep[len(dep)-1] + 2)
    while wid[len(wid)-1] * rmin < w:
        wid.append(wid[len(wid)-1] + 2)
    for i in range(0, len(dep)):
        dep[i] = dep[i] * rmin
    for i in range(0, len(wid)):
        wid[i] = wid[i] * rmin

    point = []
    for i in wid:
        for j in dep:
            point.append([i, j])
    depth = []
    width = []
    for p in point:
        depth.append(-p[1])
        width.append(p[0])

    length = []
    for i in width:
        length.append(barrier_l)

    for i in range(len(depth)):
        print(width[i], length[i], depth[i])
    m = len(length)
    ax.scatter(length, width, depth, c='red', label='destination')


def plot_hexagon_positions(barrier_l):
    odd_col_dep = [0]
    odd_col_wid = [0]
    even_col_dep = []
    even_col_wid = []
    while (odd_col_dep[-1] + np.sqrt(3) * rmin/2 <= d):
        next_dep = odd_col_dep[-1] + np.sqrt(3) * rmin
        odd_col_dep.append(next_dep)
    while (odd_col_wid[-1] + 1.5 * rmin <= w):
        next_wid = odd_col_wid[-1] + 3 * rmin
        odd_col_wid.append(next_wid)

    even_col_dep.append(np.sqrt(3)/2 * rmin)
    even_col_wid.append(1.5 * rmin)
    while (even_col_dep[-1] + np.sqrt(3) * rmin/2 <= d):
        next_dep = even_col_dep[-1] + np.sqrt(3) * rmin
        even_col_dep.append(next_dep)
    while (even_col_wid[-1] + 1.5 * rmin <= w):
        next_wid = even_col_wid[-1] + 3 * rmin
        even_col_wid.append(next_wid)
    point = []
    for i in odd_col_wid:
        for j in odd_col_dep:
            point.append([i, j])
    for i in even_col_wid:
        for j in even_col_dep:
            point.append([i, j])
    depth = []
    width = []
    for p in point:
        depth.append(-p[1])
        width.append(p[0])

    length = []
    for i in width:
        length.append(barrier_l)

    for i in range(len(depth)):
        print(width[i], length[i], depth[i])
    m = len(length)
    print("Number of sensors used:", m)
    ax.scatter(length, width, depth, c='red', label='destination')


verts = [
    [1000, 0, 0],
    [1000, 0, -2000],
    [1000, 1500, -2000],
    [1000, 1500, 0]
]

# Convert the vertices to the format required by Poly3DCollection
# Poly3DCollection expects a list of polygons, each defined by a list of vertices
verts = [verts]

# Create the polygon
# alpha for transparency
# poly = Poly3DCollection(verts, color='red', alpha=0.2)

# Add the polygon to the axis
# ax.add_collection3d(poly)
'''
'''
'''
for i in range (0, len(depth_sensor)):
    color = (0,0,1,1)
# Drasw sensors' sensing volume
    center = (length_sensor[i], width_sensor[i], depth_sensor[i]) 
    r = rmin

    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)

    x_r = r * np.sin(theta) * np.cos(phi) + center[0]
    y_r = r * np.sin(theta) * np.sin(phi) + center[1]
    z_r = r * np.cos(theta) + center[2]

    ax.plot_surface(x_r, y_r, z_r, facecolor = color, alpha = 0.1, linewidth = 0)
'''

#plot_hexagon_positions(1000)
# plot_traditional_hexagon()
plot_sensors_positions()
#ax.legend()
ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)
ax.set_zlim(-1500, 0)
ax.set_xlabel('Length')
ax.set_ylabel('Width')
ax.set_zlabel('Depth')


plt.show()
