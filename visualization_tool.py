import numpy as np
from tifffile import imread
import plotly.graph_objs as go
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage import morphology

# Path to .tif files
# ndn_tif_file = "examples/16_cell_stage_ndn.tif"
# nsn_tif_file = "examples/16_cell_stage_nsn.tif"

#ndn_tif_file = "kaggle/working/ndn/ndn_Emb2_t201.tif"
ndn_tif_file = "kaggle/working/ndn/ndn_Emb3_t001.tif"
nsn_tif_file = "kaggle 2/working/nsn/nsn_Emb3_t001.tif"


# Read the .tif file
nsn_image_stack = imread(nsn_tif_file)
ndn_image_stack = imread(ndn_tif_file)

distance = ndi.distance_transform_edt(nsn_image_stack)
markers = morphology.label(ndn_image_stack)
watershed_image = watershed(-distance, markers, mask=nsn_image_stack)

num_images = watershed_image.shape[0]
height, width = watershed_image.shape[1:]
#print(watershed_image.shape)

# Print number of segmented areas and the number of pixels
#unique_values, counts = np.unique(watershed_image, return_counts=True)
#counts_dict = dict(zip(unique_values, counts))
#for value, count in counts_dict.items():
#    print(f"Value: {value}, Count: {count}")

colors = [
    'rgb({}, {}, {})'.format(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for
    _ in range(np.max(watershed_image))]

# Create the figure
fig = go.Figure()

# Iterate through each segmented cell
for cell_value in range(1, np.max(watershed_image) + 1):
    # Mask for the current cell
    mask = (watershed_image == cell_value)

    # Extract coordinates of the current cell
    x, y, z = np.where(mask)

    # Add a scatter3d trace for the current cell
    fig.add_trace(go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=10,
            color=colors[cell_value - 1],  # Assign color to the current cell
            opacity = 1
        ),
        name=f'Cell {cell_value}'  # Label for the legend
    ))


# Set layout properties
fig.update_layout(
    title='3D Segmented Cells',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='cube'  # Ensure equal aspect ratio
    )
)

# Show the figure
fig.show()