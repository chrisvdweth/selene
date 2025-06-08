import matplotlib.pyplot as plt
import numpy as np


def visualize_binary_encodings(embed_size=6, max_pos=50, pos=None, xticks_fontsize=14, yticks_fontsize=20):
    plt.figure()

    y_offset = 0
    factor = 0

    xs = np.arange(max_pos)
    yticks_pos = []
    all_ys = []
    
    for dim in range(embed_size):
        zeros =  np.asarray([0]*(2**factor))
        ones   = np.asarray([1]*(2**factor))
        values = np.concatenate((zeros, ones), axis=0)
        # Use tiling to repeat values
        ys = np.tile(values, int(max_pos/(factor+1)))
        # Cut down array if it is too long
        if ys.shape[0] > max_pos:
            ys = ys[:max_pos]
        ys = ys.astype(np.float32)
        # Add offset to all y values
        ys += y_offset
        # Remember middle position between min/max; required to place yticks
        yticks_pos.append(y_offset+0.5)
        all_ys.append(ys)
        # Plot the step functions
        plt.step(xs, ys, c="blue")
        # Update y offset and "repeat factor"
        y_offset -= 1.5
        factor += 1

    # If no example position is given, pick a random one
    if pos is None:
        rnd_pos = np.random.randint(0, max_pos)
    else:
        rnd_pos = int(pos)

    # Draw vertical line at shown position
    plt.axvline(x=rnd_pos, color='k', linestyle='--')

    # Show label indicated the example position
    plt.xticks([rnd_pos], [f"Position {rnd_pos}"], fontsize=xticks_fontsize)

    # Generate yticks to reflect the binary representation of example position
    yticks_label = []
    for dim in range(embed_size):
        yticks_label.append(int(all_ys[dim][rnd_pos] + ((dim)*1.5)))
    plt.yticks(yticks_pos, yticks_label, fontsize=yticks_fontsize)
    
    plt.show()


def visualize_sinusoidal_encodings(embed_size=6, max_pos=50, pos=None, xticks_fontsize=14, yticks_fontsize=20):
    plt.figure()
    x_factor = 5
    x_resolution = 0.1    
    y_offset = 0
    factor = 0

    xs = np.arange(0, max_pos*x_factor, x_resolution)
    yticks_pos = []
    all_ys = []
    
    for dim in range(embed_size):
        label = ""
        xs_dim = xs
        if dim % 2 == 0:
            ys = np.sin(xs_dim / (10000**((2*dim)/16)))
            color = "blue"
            if dim == 0:
                label = "sine"
        else:
            ys = np.cos(xs_dim / (10000**((2*(dim-1))/16)))
            color = "red"
            if dim == 1:
                label = "cosine"
            #continue
        # Use tiling to repeat values
        #ys = np.tile(values, int(max_pos/(factor+1)))
        # Cut down array if it is too long
        if ys.shape[0] > (max_pos*x_factor)/x_resolution:
            ys = ys[:(max_pos*x_factor)/x_resolution]
        ys = ys.astype(np.float32)
        # Add offset to all y values
        ys += y_offset
        # Remember middle position between min/max; required to place yticks
        yticks_pos.append(y_offset+0.5)
        all_ys.append(ys)
        # Plot the step functions
        plt.plot(xs, ys, c=color, label=label)
        # Update y offset and "repeat factor"
        y_offset -= 2.5
        factor += 1

    # If no example position is given, pick a random one
    if pos is None:
        rnd_pos = np.random.randint(0, max_pos)
    else:
        rnd_pos = int(pos)
    
    # Draw vertical line at shown position
    plt.axvline(x=rnd_pos*x_factor, color='k', linestyle='--')

    # Show label indicated the example position
    plt.xticks([rnd_pos*x_factor], [f"Position {int(rnd_pos)}"], fontsize=xticks_fontsize)

    # Generate yticks to reflect the binary representation of example position
    yticks_label = []
    for dim in range(embed_size):
        val = (all_ys[dim][int((rnd_pos*x_factor)/x_resolution)] + ((dim)*2.5))
        yticks_label.append(np.round(val, decimals=3))
    plt.yticks(yticks_pos, yticks_label, fontsize=yticks_fontsize)

    plt.legend(loc="lower left")
    
    plt.show()