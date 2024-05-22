import re

pattern = re.compile(
    r"(\((?:False|True), [0-9]+\.[0-9]+, [0-9]+\.[0-9]+, [0-9]+\.[0-9]+\),\((?:False|True), [0-9]+\.[0-9]+, [0-9]+\.[0-9]+, [0-9]+\.[0-9]+\))"
)

f1 = open("in_distribution.txt").read()

r = []

for g in pattern.findall(f1):
    t1, t2 = g.split("),(")
    s, x, y, z = map(eval, t1[1:].split(","))
    s2, x2, y2, z2 = map(eval, t2[:-1].split(","))

    r.append((s, x, y, z))
    # r.append((s2,x2,y2,z2))

f1 = open("out_of_distribution.txt").read()

for g in pattern.findall(f1):
    t1, t2 = g.split("),(")
    s, x, y, z = map(eval, t1[1:].split(","))
    s2, x2, y2, y2 = map(eval, t2[:-1].split(","))
    r.append((s, x, y, z))
    # r.append((s2,x2,y2,z2))

true_data = [point[1:] for point in r if point[0]]
false_data = [point[1:] for point in r if not point[0]]

from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt

# Plot the data points
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# True data points in green
for point in true_data:
    ax.scatter(point[0], point[1], point[2], color="green")

# False data points in red
for point in false_data:
    ax.scatter(point[0], point[1], point[2], color="red")


# Define the vertices of the cube
vertices = [
    [0.5, 0, 0],
    [10.5, 0, 0],
    [10.5, 1, 0],
    [0.5, 1, 0],  # Bottom face
    [0.5, 0, 0.1],
    [10.5, 0, 0.1],
    [10.5, 1, 0.1],
    [0.5, 1, 0.1],  # Top face
]

# Define the edges connecting the vertices
edges = [
    [vertices[0], vertices[1]],
    [vertices[1], vertices[2]],
    [vertices[2], vertices[3]],
    [vertices[3], vertices[0]],  # Bottom face edges
    [vertices[4], vertices[5]],
    [vertices[5], vertices[6]],
    [vertices[6], vertices[7]],
    [vertices[7], vertices[4]],  # Top face edges
    [vertices[0], vertices[4]],
    [vertices[1], vertices[5]],
    [vertices[2], vertices[6]],
    [vertices[3], vertices[7]],  # Side edges
]

# Plot each edge
for edge in edges:
    ax.plot3D(*zip(*edge), color="b")


# Setting labels
ax.set_xlabel("Mass")
ax.set_ylabel("Friction")
ax.set_zlabel("Elasticitiy")

# ax.view_init(elev=90, azim=0) # hide z
# ax.view_init(elev=0, azim=0) # hide y
# ax.view_init(elev=0, azim=90) # hide x

plt.show()
