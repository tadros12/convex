from scipy.spatial import ConvexHull
import numpy as np
from ucimlrepo import fetch_ucirepo 
import plotly.express as px

import cvxpy


iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.target

print(X.shape)

def is_convex_dataset(points):
    hull = ConvexHull(points)
    return np.array_equal(points, points[hull.vertices])

def convexication(points):
    hull = ConvexHull(points)
    return points[hull.vertices]
plot = px.scatter(x=X.values[:, 0], y=X.values[:, 1], color=y)
plot.show()

print("is the data set convex?:", is_convex_dataset(X.values[:, :2]))

convex_hull_points = convexication(X.values[:, :2])


plot = px.scatter(x=X.values[:, 0], y=X.values[:, 1], color=y)
plot.add_scatter(x=convex_hull_points[:, 0], y=convex_hull_points[:, 1], mode='lines+markers', name='Convex Hull')
plot.show()


print("is the data set convex?:", is_convex_dataset(convex_hull_points[:, :2]))
