import matplotlib.pyplot as plt
from shapely.geometry import Polygon

"""
This script was used to test the 2D IoU calculation.

Conclusion: Points from bounding box corners were not in the right order.
"""

## (Polygon coördinates that resulted from the projected oriented bounding box from O3D)
coords1 = [
    (382.9125900472306, 703.5409221510969),
    (520.4405921327161, 665.3418588955216),
    (352.9310742315926, 607.9908538177757),
    (490.4590763170782, 569.7917905622004)
]

coords2 = [
    (381.1543839713567, 704.3168853702317),
    (492.26592205832407, 664.1517973703993),
    (350.62374279595207, 621.2299014218469),
    (461.7352808829194, 581.0648134220145)
]

# Split into x and y
x1, y1 = zip(*coords1)
x2, y2 = zip(*coords2)

plt.figure(figsize=(10, 6))

# Plot points
plt.scatter(x1, y1, color='blue', label='Coords 1', s=100, marker='o')
plt.scatter(x2, y2, color='red', label='Coords 2', s=100, marker='^')

# Annotate points
for i, (x, y) in enumerate(coords1):
    plt.text(x + 2, y + 2, f'P1-{i}', color='blue')
for i, (x, y) in enumerate(coords2):
    plt.text(x + 2, y + 2, f'P2-{i}', color='red')

plt.title("Raw Polygon Points Before Shapely Conversion")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()

# Create shapely polygons
poly1 = Polygon(coords1)
poly2 = Polygon(coords2)

poly1 = poly1.buffer(0)
poly2 = poly2.buffer(0)

# Extract exterior coordinates
x1, y1 = poly1.exterior.xy
x2, y2 = poly2.exterior.xy

# Plotting
plt.figure()
plt.fill(x1, y1, color='blue', alpha=0.5, label='Polygon 1')
plt.fill(x2, y2, color='red', alpha=0.5, label='Polygon 2')
plt.legend()
plt.title("Debugging Two Polygons")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")  # To maintain aspect ratio
plt.grid(True)
plt.show()
