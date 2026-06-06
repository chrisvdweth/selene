import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def plot_data_2d(X, X_pca=None, pcs=None, x_label="$x_1$", y_label="$x_2$", size=50, max_lim=3):

    plt.figure()
    #plt.axis('equal')
    plt.gca().set_aspect('equal')
    plt.xlim(-max_lim,max_lim)
    plt.ylim(-max_lim,max_lim)

    plt.scatter(X[:,0], X[:,1], s=size, clip_on=False, color='blue', alpha=0.6)

    if X_pca is not None:
        X_pca = np.asarray(X_pca)
        for i in range(X.shape[0]):
            s, t = X[i], X_pca[i]
            plt.plot((s[0], t[0]), (s[1], t[1]), '--', c='black', lw=0.5)
        plt.scatter(X_pca[:,0], X_pca[:,1], s=size, clip_on=False, c='red', alpha=0.3)
        #plt.plot(X_pca[:,0], X_pca[:,1], clip_on=False, c='red')
        plt.axline(X_pca[0], X_pca[1], color='red', linestyle='--')

    if pcs is not None:
        pcs = np.asarray(pcs)
        for pc in pcs.T:
            plt.arrow(0, 0, *pc, head_width=0.15, head_length=0.2, color='black', lw=3)   
    
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.show()



def plot_data_3d(X, pca=None, show_projection=False, show_mapping=False, show_pcs=0, pcs_scaling=0.5, x_label="$x_1$", y_label="$x_2$", z_label="$x_3$", point_size=50, alpha=0.5, elev=20, max_lim=2, azim=None, repeat=False):

    X_pca = None
    if pca is not None:
        X_pca = pca.transform(X)
        X_pca = pca.inverse_transform(X_pca)
        pcs = pca.components_.T * pca.explained_variance_ * pcs_scaling
    
    X = np.asarray(X)

    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Set the limits for x, y, and z axes
    ax.set_xlim(-max_lim, max_lim)
    ax.set_ylim(-max_lim, max_lim)
    ax.set_zlim(-max_lim, max_lim)


    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=point_size, alpha=alpha)

    if X_pca is not None and show_projection is True:
        # Get the bounding box of your data to limit the plane size appropriately
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        if pca.n_components == 1:
            ### Line
            direction = X_pca[0] - X_pca[-1]
            # 2. Define a massive range for 't' to simulate "infinity"
            # t=0 is p1, t=1 is p2. t=-100 to 100 goes way beyond them.
            t = np.linspace(-x_min, x_min, 2)[:, np.newaxis]
            # 3. Calculate the infinite line coordinates using: r(t) = p0 + t*v
            line_points = X_pca[0] + t * direction  # Result is a 2x3 matrix containing both endpoints
            # To plot it:
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color='red', linestyle='--', alpha=alpha)
        elif pca.n_components == 2:
            ### Plane
            # Subtract the mean to center the data
            centroid = X_pca.mean(axis=0)
            X_centered = X_pca - centroid
            # Run SVD
            _, _, Vh = np.linalg.svd(X_centered)
            # The last row of Vh (or V transpose) is the normal vector (A, B, C)
            normal = Vh[-1]
            A, B, C = normal
            # D is calculated using the centroid: D = - (A*x0 + B*y0 + C*z0)
            D = -np.dot(normal, centroid)
            # Generate a grid spanning the data limits
            x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 10),  np.linspace(y_min, y_max, 10))
            # Calculate Z on the grid: z = (-A*x - B*y - D) / C
            z_grid = (-A * x_grid - B * y_grid - D) / C
            # Plot the infinite surface
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.1, color='red', edgecolor='none')
            
        if show_mapping is True:
            for i in range(X.shape[0]):
                s, t = X[i], X_pca[i]
                plt.plot((s[0], t[0]), (s[1], t[1]), (s[2], t[2]), '--', c='black', lw=1, alpha=alpha)
            
        ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], s=point_size, clip_on=False, c='red', alpha=alpha)
        
    if X_pca is not None and show_pcs > 0:
        pcs = np.asarray(pcs).T
        for i in range(show_pcs):
            pc = pcs[i]
            ax.quiver(0, 0, 0, *pc, color="black", lw=3, arrow_length_ratio=0.1)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    def update(frame):
        ax.view_init(elev=elev, azim=frame)
        return []

    if azim is None:
        anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, repeat=repeat)
        plt.tight_layout()
        plt.close(fig)
        return HTML(anim.to_html5_video())
    else:
        update(azim)
        plt.tight_layout()            
        plt.show()


def plot_explained_variance(values, style="bar", show_ratios=False):
    if len(values) == 0:
        raise ValueError("values must contain at least one element")

    max_val = np.max(values)
    
    xs = range(len(values))

    fig, ax = plt.subplots(figsize=(6, 4))

    if style == "line":
        ax.plot(xs, values)
        if show_ratios is True:
            for x, value in zip(xs, values):
                ax.text(x, value, f"{value*100:.3g}%", ha="center", va="bottom", fontsize=12)
    elif style == "bar":
        bars = ax.bar(xs, values)
        if show_ratios is True:
            # Add value labels above each bar
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value*100:.3g}%", ha="center", va="bottom", fontsize=12)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"PC{str(i+1)}" for i in xs])
    ax.set_xlabel("Principal Component", fontsize=14)
    ax.set_ylabel("Explained Variance", fontsize=14)
    ax.set_ylim(0, max_val*1.2)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    plt.show()