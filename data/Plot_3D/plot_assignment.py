import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


f = open('../../Data/Uniform_Distribution/150_sensors/150_10.txt', 'r')


depth_sensor = []
width_sensor = []
length_sensor = []
n = int(f.readline())
for i in range (0, n):
    s = f.readline()
    s = s.split(' ')
    depth_sensor.append(-(float)(s[2]))
    length_sensor.append((float)(s[1]))
    width_sensor.append((float)(s[0]))
ax = plt.figure().add_subplot(projection='3d')
#ax.scatter(length_sensor, width_sensor, depth_sensor, c = 'blue', label = 'sensors')

rmin = 84.7
w = 2000
d = 1500
dep = [0]
wid = [0]

while dep[len(dep)-1] * rmin < d:
    dep.append(dep[len(dep)-1] + 2)
while wid[len(wid)-1] * rmin < w:
    wid.append(wid[len(wid)-1] + 2)
for i in range (0, len(dep)):
    dep[i] = dep[i] * rmin
for i in range (0, len(wid)):
    wid[i] = wid[i] * rmin


def plot_barrier(L0):
    point = []
    for i in wid:
        for j in dep:
            point.append([i,j])
    depth = []
    width = []
    for i in point:
        depth.append(-i[1])
        width.append(i[0])

    length = []
    for i in width:
        length.append(L0)

    for i in range (len(depth)):
        print(width[i], length[i], depth[i])
    m = len(length)
    ax.scatter(length, width, depth, c='red')

    verts = [
    [L0, 0, 0],
    [L0, 0, -1500],
    [L0, 2000, -1500],
    [L0, 2000, 0]
    ]

    # Convert the vertices to the format required by Poly3DCollection
    verts = [verts]  # Poly3DCollection expects a list of polygons, each defined by a list of vertices

    # Create the polygon
    poly = Poly3DCollection(verts, color='red', alpha=0.2)  # alpha for transparency

    # Add the polygon to the axis
    ax.add_collection3d(poly)

plot_barrier(0)
plot_barrier(1000)
plot_barrier(2000)

def plot_assignment(length, width, depth, length_sensor, width_sensor, depth_sensor):
    assign = [0]
    for i in range (1, m+1):
        sensor = int(input())
        assign.append(sensor-1)

    ax.plot([length[0], length_sensor[assign[1]]], [width[0], width_sensor[assign[1]]], [depth[0], depth_sensor[assign[1]]], c = 'purple', label = 'Assignment')
    for i in range (2, m+1):
        ax.plot([length[i-1], length_sensor[assign[i]]], [width[i-1], width_sensor[assign[i]]], [depth[i-1], depth_sensor[assign[i]]], c = 'purple')




#ax.legend()
ax.set_xlim(0, 2000)
ax.set_ylim(0, 2000)
ax.set_zlim(-1500, 0)
ax.set_xlabel('Length')
ax.set_ylabel('Width')
ax.set_zlabel('Depth')


plt.show()
