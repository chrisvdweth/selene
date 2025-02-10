import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from matplotlib import cm

color_cycle = cycle('bgrcmyk')



def plot_csi_data(X, y, regression_lines=[], show_errors=False):
    plt.figure()
    plt.scatter(X, y, color='blue', alpha=0.6, label="data points")

    for idx, (w, color, label) in enumerate(regression_lines):
        if color is None:
            color = next(color_cycle)
        plt.gca().axline(xy1=(0, w[0]), slope=w[1], color=color, lw=2, label=label)            
        regression_line = [ (w[0] + w[1]*x) for x in X ]

    if len(regression_lines) > 0 and show_errors is True:
        for i in range(len(X)):
            plt.plot([X[i:i+1], X[i:i+1]], [y[i:i+1], regression_line[i:i+1]], '--', c="black")

    plt.xlim([X.min()*0.95, X.max()*1.05])
    plt.ylim([y.min()*0.95, y.max()*1.05])
    plt.legend(loc='lower right')
    plt.xlabel("Shoe Print Size (cm)")
    plt.ylabel("Height (cm)")
    plt.show()


def plot_random_losses(xs, ys, zs):
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(30, -30)
    ax.set_xlabel(r'$w_1$', fontsize=16)
    ax.set_ylabel(r'$w_0$', fontsize=16)
    ax.set_zlabel('L', fontsize=16)
    ax.scatter(xs, ys, zs)
    plt.tight_layout()
    plt.show()


def plot_loss_function(X, y, w0_range, w1_range, loss_func):
    Xs, Ys = np.meshgrid(w0_range, w1_range)
    Zs = np.zeros_like(Xs)
    for i0, w0 in enumerate(w0_range):
        for i1, w1 in enumerate(w1_range):
            loss = loss_func(X, y, [w0, w1])
            Zs[i1,i0] = loss
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(30, -30)
    ax.set_xlabel(r'$w_1$', fontsize=16)
    ax.set_ylabel(r'$w_0$', fontsize=16)
    ax.set_zlabel('L', fontsize=16)
    surf = ax.plot_surface(Xs, Ys, Zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.tick_params(labelsize=10)
    plt.tight_layout()
    plt.show()



def plot_truth_vs_predictions(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6)
    plt.gca().axline((1, 1), slope=1, c='r', lw=3)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    max_val = np.maximum.reduce([y_true, y_pred], axis=(1,0))
    min_val = np.minimum.reduce([y_true, y_pred], axis=(1,0))
    plt.xlim([min_val*0.9, max_val*1.1])
    plt.ylim([min_val*0.9, max_val*1.1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_anscombes_quartet(datasets, show_regression_line=False):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    axs[0, 0].set(xlim=(0, 20), ylim=(2, 14))
    axs[0, 0].set(xticks=(0, 10, 20), yticks=(4, 8, 12))

    for ax, (label, (x, y)) in zip(axs.flat, datasets.items()):
        ax.text(0.1, 0.9, label, fontsize=20, transform=ax.transAxes, va='top')
        ax.tick_params(direction='in', top=True, right=True)
        ax.plot(x, y, 'o')
    
        # linear regression
        if show_regression_line == True:
            w1, w0 = np.polyfit(x, y, deg=1)  # slope, intercept
            ax.axline(xy1=(0, w0), slope=w1, color='r', lw=2)    
        
        # add text box for the statistics
        stats = (f'$\\mu$ = {np.mean(y):.2f}\n'
                 f'$\\sigma$ = {np.std(y):.2f}\n'
                 f'$r$ = {np.corrcoef(x, y)[0][1]:.2f}')
        bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        ax.text(0.95, 0.07, stats, fontsize=9, bbox=bbox,transform=ax.transAxes, horizontalalignment='right')
    
    plt.show()


def plot_huber_loss(deltas=[3], show_mae=False, show_mse=False):
    x = np.linspace(-5, 5, 1000)
    plt.figure()
    for d in deltas:
        y_mse = 0.5*np.square(x)
        y_mae = d*np.abs(x) - 0.5*np.square(d)
        mae_range = np.argwhere(np.abs(x) > d)
        mse_range = np.argwhere(np.abs(x) < d)
        y_mae[mse_range] = 0
        y_mse[mae_range] = 0
        #print(y_mae)
        y = y_mse + y_mae
        plt.plot(x, y, linewidth=3, label=f"$\delta$={d}");
    if show_mae is True:
        plt.plot(x, np.abs(x), linewidth=3, label="MAE", linestyle='--');
    if show_mse is True:
        plt.plot(x, np.square(x), linewidth=3, label="MSE", linestyle='--');
    plt.legend(loc="upper center")
    plt.show()


def plot_correlation_matrix(correlation_matrix, cmap="coolwarm", font_size=12):
    plt.figure()
    with plt.style.context({'axes.labelsize':12, 'xtick.labelsize':font_size, 'ytick.labelsize':font_size}):
        ax = sns.heatmap(correlation_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, square=True, annot=True, annot_kws={'size': font_size})
    plt.show()
