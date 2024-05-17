from PIL import Image
import os

# Directory containing the images
directory = 'results_test'

# List to hold images
images = []

# Iterate through files in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".png") and "_bw" in filename:
        # Construct full file path
        file_path = os.path.join(directory, filename)
        # Open and append the image to the list
        images.append(Image.open(file_path))

# Create a GIF
if images:
    # Save the first image and append the rest
    gif_path = os.path.join(directory, 'output_batch.gif')
    images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
    print("GIF created successfully at:", gif_path)
else:
    print("No suitable images found for creating GIF.")

