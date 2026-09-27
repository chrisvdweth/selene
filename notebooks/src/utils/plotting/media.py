import numpy as np
import matplotlib.pyplot as plt

def show_image_from_array(image_array, channel=None, pixel_size=1, dpi=100, interpolation='nearest', show_axis=False):
    #plt.figure()
    height, width = image_array.shape[:2]
    fig = plt.figure(figsize=(width * pixel_size / dpi,
                              height * pixel_size / dpi),
                     dpi=dpi)

    # Make the axes fill the entire figure
    ax = fig.add_axes([0, 0, 1, 1])

    if channel is not None and channel < image_array.shape[-1]:
        channel_image = np.zeros_like(image_array)
        channel_image[:, :, channel] = image_array[:, :, channel]
        ax.imshow(channel_image, interpolation='nearest')
    else:
        ax.imshow(image_array, interpolation='nearest')
    if show_axis is False:
        ax.axis("off")
    plt.show()    