import numpy as np
from PIL import Image

# Parameters
map_size_m = 10.0  # 10 meters square
resolution = 0.05  # meters per pixel
map_pixels = int(map_size_m / resolution)  # 200 pixels
thickness_m = 0.1  # obstacle thickness 10 cm
thickness_px = int(thickness_m / resolution)  # 2 pixels

# Create empty map: all free space = 254
map_array = np.full((map_pixels, map_pixels), 254, dtype=np.uint8)

# Draw hollow square obstacle (black = 0)
# Draw outer border 2 pixels thick black
map_array[:thickness_px, :] = 0  # Top border
map_array[-thickness_px:, :] = 0  # Bottom border
map_array[:, :thickness_px] = 0  # Left border
map_array[:, -thickness_px:] = 0  # Right border

# Save as PGM
img = Image.fromarray(map_array)
img.save('map.pgm')

# Write YAML file
yaml_content = f"""
image: map.pgm
resolution: {resolution}
origin: [0.0, 0.0, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""

with open('map.yaml', 'w') as f:
    f.write(yaml_content.strip())

print("Map files 'map.pgm' and 'map.yaml' created.")
