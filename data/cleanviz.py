import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def process_cfd_data(filepath, resolution=(128, 128)):
    # Load data
    data = np.loadtxt(filepath)

    # Points where data is known
    points = data[:, :2]  # considering only x, y coordinates

    # Grid points where we want to interpolate data
    grid_x, grid_y = np.mgrid[np.min(points[:, 0]):np.max(points[:, 0]):complex(resolution[0]), np.min(points[:, 1]):np.max(points[:, 1]):complex(resolution[1])]

    # Interpolate pressure
    grid_p = griddata(points, data[:, 3], (grid_x, grid_y), method='cubic')

    # Interpolate U_x and U_y
    grid_Ux = griddata(points, data[:, 4], (grid_x, grid_y), method='cubic')
    grid_Uy = griddata(points, data[:, 5], (grid_x, grid_y), method='cubic')

    # Plotting for visualization
    plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(grid_p, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower')
    plt.title('Pressure')
    plt.setp(ax1, xticks=[], yticks=[])  # Remove x and y labels

    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(grid_Ux, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower')
    plt.title('Velocity U_x')
    plt.setp(ax2, xticks=[], yticks=[])  # Remove x and y labels

    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(grid_Uy, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()), origin='lower')
    plt.title('Velocity U_y')
    plt.setp(ax3, xticks=[], yticks=[])  # Remove x and y labels

    # Adjust subplot spacing more closely
    plt.subplots_adjust(wspace=0.1)  # Adjust the width space between subplots to make them closer

    plt.show()

    return grid_p, grid_Ux, grid_Uy

# Call the function with your file path
# grid_p, grid_Ux, grid_Uy = process_cfd_data('path_to_your_data.txt')
process_cfd_data('/home/melkor/projects/fluid_stuff/Deep-Flow-Prediction/data/OpenFOAM/postProcessing/internalCloud/500/cloud.xy')

