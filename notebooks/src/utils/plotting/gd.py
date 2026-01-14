import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML



def plot_function(f):
    xs = np.linspace(-1, 5, 400)
    ys = f(xs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(0, 12)
    ax.set_xlim(-1.5, 5.5)
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.plot(xs, ys, label="f(x)", c="blue", linewidth=3, alpha=0.25)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=16)
    plt.tight_layout()
    plt.show()


def animate_gradient(f, df, x_start=0, x_end=1, step=0.5, repeat=False, show_step=None):
    xs = np.linspace(-1.1, 5.1, 400)
    ys = f(xs)

    x_vals = np.arange(x_start, x_end+step, step)
    y_vals = [ f(x) for x in x_vals ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(0, 12)
    ax.set_xlim(-1.5, 5.5)
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.plot(xs, ys, label="f(x)", c="blue", linewidth=3, alpha=0.25)
    
    ax.set_title(f"Gradient:")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=16)

    point = ax.scatter([], [], color='blue', s=25, alpha=0.5)
    slope, = ax.plot([], [], c="black", linewidth = 2, label="slope", linestyle='dashed')
    
    def update(frame):
        x, y = x_vals[frame], y_vals[frame]
        gradient = df(x)
        tangent_range = np.linspace(x-1, x+1, 10)
        point.set_offsets([x, y])
        slope.set_data(tangent_range, (gradient*(tangent_range - x) + y))
        ax.set_title(f"Gradient at x={x}: {gradient}")

    if show_step is None:
        anim = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=500, repeat=repeat)
        plt.close(fig)
        return HTML(anim.to_html5_video())
    else:
        if show_step > len(xs):
            print("[Error] Valuer of show_step is too large.")
            return
        for i in range(show_step+1):
            update(i)
        plt.tight_layout()
        plt.show()  


# Gradient Descent core
def gd(f, df, x0, eta, n_steps):
    xs = [x0]
    ys = [f(x0)]
    x = x0
    for _ in range(n_steps):
        x = x - eta*df(x)
        xs.append(x)
        ys.append(f(x))
    return xs, ys



def animate_gradient_descent(f, df, x0=0, eta=0.1, n_steps=20, repeat=False, show_step=None):
    xs = np.linspace(-1.1, 5.1, 400)
    ys = f(xs)

    x_vals, y_vals = gd(f, df, x0, eta, n_steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(0, 12)
    ax.set_xlim(-1.5, 5.5)
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.plot(xs, ys, label="f(x)", c="blue", linewidth=3, alpha=0.25)
    
    ax.set_title(f"Gradient Descent ($\eta$ = {eta})")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=16)
    
    def update(frame):
        x, y = x_vals[frame], y_vals[frame]
        ax.scatter(x, y, color='blue', s=25, alpha=0.5)
        if frame > 0:
            x_prev, y_prev = x_vals[frame-1], y_vals[frame-1]
            ax.quiver([x_prev], [y_prev], [x-x_prev], [y-y_prev], scale_units='xy', angles='xy', scale=1)
        ax.set_title(f"Gradient Descent (η = {eta}), Step: {frame}")

    if show_step is None:
        anim = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=500, repeat=repeat)
        plt.close(fig)
        return HTML(anim.to_html5_video())
    else:
        if show_step > n_steps:
            print("[Error] n_steps must be larger than show_step.")
            return
        for i in range(show_step+1):
            update(i)
        plt.tight_layout()            
        plt.show()


def animate_gradient_descent_nonconvex(f, df, x0=0, eta=0.1, n_steps=20, repeat=False, show_step=None):
    xs = np.linspace(-5.1, 14.1, 400)
    ys = f(xs)

    x_vals, y_vals = gd(f, df, x0, eta, n_steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(-1, 23)
    ax.set_xlim(-5.5, 14.5)
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel("y", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.plot(xs, ys, label="f(x)", c="blue", linewidth=3, alpha=0.25)
    
    ax.set_title(f"Gradient Descent ($\eta$ = {eta})")
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=16)
    
    def update(frame):
        x, y = x_vals[frame], y_vals[frame]
        ax.scatter(x, y, color='blue', s=25, alpha=0.5)
        if frame > 0:
            x_prev, y_prev = x_vals[frame-1], y_vals[frame-1]
            ax.quiver([x_prev], [y_prev], [x-x_prev], [y-y_prev], scale_units='xy', angles='xy', scale=1)
        ax.set_title(f"Gradient Descent (η = {eta}), Step: {frame}")

    if show_step is None:
        anim = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=500, repeat=repeat)
        plt.close(fig)
        return HTML(anim.to_html5_video())
    else:
        if show_step > n_steps:
            print("[Error] n_steps must be larger than show_step.")
            return
        for i in range(show_step+1):
            update(i)
        plt.tight_layout()
        plt.show()





def plot_multivariate_function(f, resolution=100):

    # Create grid
    x1= np.linspace(-10, 10, resolution)
    x2 = np.linspace(-4,-4, resolution)
    X1, X2 = np.meshgrid(x1, x1)

    # Compute Z values
    Y = f(X1, X2)

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)
    
    # Plot surface
    surf = ax.plot_surface(X1, X2, Y, cmap="Blues", edgecolor="none")

    # Labels
    ax.set_xlabel("x$_1$", fontsize=16)
    ax.set_ylabel("x$_2$", fontsize=16)
    ax.set_zlabel("y", fontsize=16)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()



def gd_multivar(f, df, x1, x2, eta, n_steps):
    xs = [(x1, x2)]
    for _ in range(n_steps):
        g1, g2 = df(x1, x2)
        x1, x2 = x1 - eta*g1, x2 - eta*g2
        xs.append((x1, x2))
    return np.asarray(xs)


def plot_gradient_descent_multivariate(f, df, x0=(0, 0), eta=0.1, n_steps=20, repeat=False, show_step=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_ylim(0, 12)
    ax.set_xlim(-1.5, 5.5)
    ax.set_xlabel("x$_1$", fontsize=18)
    ax.set_ylabel("x$_2$", fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    x_vals = gd_multivar(f, df, x0[0], x0[1], eta, n_steps)
    
    # Contours
    x1s = np.linspace(-10, 10, 400)
    x2s = np.linspace(-4, 4, 400)
    X1, X2 = np.meshgrid(x1s, x2s)
    Y = f(X1, X2)
    ax.contour(X1, X2, Y, levels=30, cmap='Blues')

    # Initial black dot
    ax.plot(x_vals[0, 0], x_vals[0, 1], 'ko', label="Start")

    # Red dot + path line
    point, = ax.plot([], [], 'ro', markersize=3)
    path, = ax.plot([], [], 'r-', linewidth=0.7)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-4, 4)
    #ax.axis('off')
    #ax.set_title(f"SGD with η = {eta}", fontsize=14)

    def update(frame):
        #ax.plot(x_vals[frame, 0], x_vals[frame, 1], 'ko', label="Start")
        path.set_data([x_vals[:frame+1, 0]], [x_vals[:frame+1, 1]])
        point.set_data([x_vals[frame, 0]], [x_vals[frame, 1]])
        ax.set_title(f"Gradient Descent (η = {eta}), Step: {frame}")

    if show_step is None:
        anim = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=100, repeat=False)
        plt.close()
        return HTML(anim.to_html5_video())
    else:
        if show_step > n_steps:
            print("[Error] n_steps must be larger than show_step.")
            return
        for i in range(show_step+1):
            update(i)
        plt.tight_layout()
        plt.show()
