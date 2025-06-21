import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def plot_data(X_train, y_train):
    X_sin = np.linspace(0, 2*np.pi, 200)
    y_sin = np.sin(X_sin).ravel()
    fig = plt.figure()
    x_min_plot, x_max_plot = np.min(X_train)-0.75, np.max(X_train)+0.75
    plt.xlim(x_min_plot, x_max_plot)
    plt.ylim(-2, 2)
    plt.plot(X_sin, y_sin, color="k", lw=1, linestyle="--", label="True function f(x)")
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label=f"Training data ({len(X_train)})")
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.legend(loc="lower left", fontsize=14)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.show()


def plot_results(X_train, y_train, poly_degree=0, color="red"):
    X_sin = np.linspace(0, 2*np.pi, 200)
    y_sin = np.sin(X_sin).ravel()
    
    poly = PolynomialFeatures(poly_degree, interaction_only=False)
    X_poly = poly.fit_transform(X_train)

    pol_reg = LinearRegression(fit_intercept=False) # PolynomialFeatures!!!
    pol_reg.fit(X_poly, y_train)
    
    x_min_plot, x_max_plot = np.min(X_sin)-0.75, np.max(X_sin)+0.75
    X_plot = np.linspace(x_min_plot, x_max_plot, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = pol_reg.predict(X_plot_poly)
    
    fig = plt.figure()
    plt.xlim(x_min_plot, x_max_plot)
    plt.ylim(-2, 2)
    
    plt.plot(X_sin, y_sin, color="k", lw=1, linestyle="--", label="True function f(x)")
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, label=f"Training data ({len(X_train)})")
    plt.plot(X_plot, y_plot, color=color, label=f'Polynomial degree {poly_degree}', lw=3)
    plt.xlabel('x', fontsize=18)
    plt.ylabel('y', fontsize=18)
    plt.legend(loc="lower left", fontsize=14)
    
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.show()


def plot_multiple_results(X_train, y_train, n_points=20, poly_degree=0, n_rounds=100, seed=None, color="red"):

    if seed is not None:
        np.random.seed(seed)

    x_min_plot, x_max_plot = 0, 2*np.pi
    
    X_sin = np.linspace(x_min_plot, x_max_plot, 200).reshape(-1, 1)
    y_sin = np.sin(X_sin).ravel()
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax1.set_xlim(x_min_plot, x_max_plot)
    ax2.set_xlim(x_min_plot, x_max_plot)
    ax1.set_ylim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    ax1.plot(X_sin, y_sin, color="k", lw=1, linestyle="--", label="True function f(x)")
    ax2.plot(X_sin, y_sin, color="k", lw=1, linestyle="--", label="True function f(x)")

    all_pred = np.zeros((n_rounds, y_sin.shape[0]))
    X_plot = np.linspace(x_min_plot, x_max_plot, 100).reshape(-1, 1)
    for i in range(n_rounds):
        indices = np.random.choice(len(X_train), size=n_points, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    
        poly = PolynomialFeatures(poly_degree, interaction_only=False)
        X_poly = poly.fit_transform(X_train_sub)
    
        model = LinearRegression(fit_intercept=False) # PolynomialFeatures!!!
        model.fit(X_poly, y_train_sub)
    
        y_pred = model.predict(poly.fit_transform(X_sin))
        all_pred[i] = y_pred
    
        X_plot_poly = poly.transform(X_plot)
        y_plot = model.predict(X_plot_poly)

        if i == 0:
            label = f"Hypothesis h(x) (p={poly_degree})"
        else:
            label = ""
            
        ax1.plot(X_plot, y_plot, color=color, alpha=0.1, label=label)
    
    g_bar = np.mean(all_pred, axis=0)
    g_std = np.std(all_pred, axis=0)
    
    ax2.fill_between(X_sin.squeeze(), g_bar+g_std, g_bar-g_std, color=color, alpha=0.2)
    ax2.plot(X_sin, g_bar, color=color, lw=2, label=f"Average hypothesis $\overline{{h}}(x)$ (p={poly_degree})")
    ax2.plot(X_sin, g_bar+g_std, color=color, alpha=0.6, lw=1)
    ax2.plot(X_sin, g_bar-g_std, color=color, alpha=0.6, lw=1)
    ax1.set_xlabel('x', fontsize=18)
    ax2.set_xlabel('x', fontsize=18)
    ax1.set_ylabel('y', fontsize=18)
    ax2.set_ylabel('y', fontsize=18)
    ax1.legend(loc="lower left", fontsize=12)
    ax2.legend(loc="lower left", fontsize=12)
    ax1.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    ax2.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.show()