import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import plotly.express as px
from scipy.spatial import ConvexHull



tickers = ['AAPL']  

start_date = '2020-01-01'
end_date = '2024-12-6'

adj_close = yf.download(tickers, start=start_date, end=end_date)['Adj Close']



adj_close_reset = adj_close.reset_index()
adj_close_melted = adj_close_reset.melt(id_vars='Date', value_vars=tickers, var_name='Stock', value_name='Adj Close')



def is_convex_dataset(points):
    hull = ConvexHull(points)
    return np.array_equal(points, points[hull.vertices])


def convexication(points):
    hull = ConvexHull(points)
    return points[hull.vertices]





adj_close_melted = adj_close_melted.reset_index()

adj_close_melted['Date_numeric'] = adj_close_melted['Date'].apply(lambda x: x.toordinal())

points = adj_close_melted[['Date_numeric', 'Adj Close']].values

is_convex = is_convex_dataset(points)
print(f"Is the dataset convex? {is_convex}")

convex_hull_points = convexication(points)
is_convex = is_convex_dataset(convex_hull_points)
print(f"is the dataset convex after convexcation ? {is_convex}")


plot = px.scatter(adj_close_melted, x='Date', y='Adj Close', color='Stock',
                  title='Adjusted Close Prices Over Time')

convex_hull_dates = [pd.Timestamp.fromordinal(int(date)) for date in convex_hull_points[:, 0]]

plot.add_scatter(x=convex_hull_dates, y=convex_hull_points[:, 1], mode='lines+markers', name='Convex Hull')
plot.show()


