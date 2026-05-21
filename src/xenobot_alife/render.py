import matplotlib.pyplot as plt


def plot_body(body, path=None):
    fig, ax = plt.subplots()
    ax.imshow(body.voxels, interpolation="nearest")
    ax.set_title("Voxel phenotype: 0 empty, 1 passive, 2 contractile, 3 propulsive")
    ax.set_xticks([]); ax.set_yticks([])
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_trajectory(metrics, path=None):
    traj = metrics["trajectory"]
    fig, ax = plt.subplots()
    ax.plot(traj[:, 0], traj[:, 1])
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Xenobot trajectory")
    if path:
        fig.savefig(path, bbox_inches="tight")
    return fig
