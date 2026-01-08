import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


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

# Animation function for a given learning rate
def animate_gradient_descent(f, df, x0=0, eta=0.1, n_steps=10, repeat=False):
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

    anim = animation.FuncAnimation(fig, update, frames=len(x_vals), interval=500, repeat=repeat)
    plt.close(fig)
    return HTML(anim.to_html5_video())