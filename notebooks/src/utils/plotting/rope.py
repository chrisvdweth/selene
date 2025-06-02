import numpy as np
import matplotlib.pyplot as plt


def show_radian_example(theta):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    
    # Draw a unit circle
    circle = plt.Circle((0, 0), 1, color='lightgray', fill=False, linestyle='--')
    ax.add_artist(circle)
    
    # Compute the arc endpoint
    x = np.cos(theta)
    y = np.sin(theta)
    
    # Draw radius lines
    #ax.plot([0, 1], [0, 0], 'b', label='Radius')         # base radius
    #ax.plot([0, x], [0, y], 'b')                         # angle radius
    plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='k', label="Input vector")   # input vector
    plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='b', label="Rotated vector") # rotated radius
    
    # Draw arc
    arc = np.linspace(0, theta, 100)
    ax.plot(np.cos(arc), np.sin(arc), 'r', lw=2, label=f"Arc length = {theta:.2f} radian")
    
    # Annotate the angle
    ax.text(0.5, 0.1, f"{theta:.2f} radian", fontsize=12, color='red')
    
    # Set plot limits and labels
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_vectors(vectors, styles=None, colors=None, labels=None, width_factor=1, legend_loc="lower right"):
    origin = np.zeros((len(vectors), 2))  # All vectors start at the origin
    vectors = np.array(vectors) # Just in case vectors is not a numpy array

    if colors is None:
        colors = ['b'] * len(vectors)

    plt.figure()

    for idx, vec in enumerate(vectors):
        try:
            label = labels[idx]
        except:
            label = ""
        try:
            label = labels[idx]
        except:
            label = ""            
        plt.quiver(0, 0, vec[0],vec[1], 
                   angles='xy', 
                   scale_units='xy', 
                   scale=1, 
                   color=colors[idx], 
                   width=width_factor*0.01, 
                   label=label)

    # Set the plot limits
    all_coords = np.vstack((origin, vectors))
    max_val = np.abs(all_coords).max() * 1.2
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    #
    plt.axhline(0, color='k', lw=1)
    plt.axvline(0, color='k', lw=1)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    if labels is not None:
        plt.legend(loc=legend_loc) 
    plt.show()    
