# src/Visualize.py
"""
PyVista 3-D voxel renderer for Xenobot genomes.

"""

import numpy as np
from src.representation import EMPTY, PASSIVE, ACTIVE_P, ACTIVE_N

MAT_COLORS = {
    PASSIVE:  [0.8, 0.8, 0.8],   # grey
    ACTIVE_P: [0.2, 0.6, 1.0],   # blue
    ACTIVE_N: [1.0, 0.3, 0.1],   # red
}

MAT_NAMES = {
    PASSIVE:  "Passive",
    ACTIVE_P: "Active+ (expands)",
    ACTIVE_N: "Active− (contracts)",
}

    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def render_genome(genome: np.ndarray,
                  title: str = "Xenobot",
                  save_path: str = None,
                  backend: str = "auto"):

    if backend == "auto":
        backend = "pyvista" if _HAS_PYVISTA else "matplotlib"

    if backend == "pyvista" and _HAS_PYVISTA:
        return _render_pyvista(genome, title, save_path)
    else:
        return _render_matplotlib(genome, title, save_path)


def render_side_by_side(genomes: list, titles: list = None, save_path: str = None):

    n = len(genomes)
    titles = titles or [f"Robot {i+1}" for i in range(n)]
    fig = plt.figure(figsize=(4 * n, 4))

    for i, (genome, t) in enumerate(zip(genomes, titles)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        _draw_voxels_matplotlib(ax, genome)
        ax.set_title(t, fontsize=9)
        ax.set_axis_off()

    _add_legend_matplotlib(fig)
    fig.suptitle("Side-by-Side Comparison", y=1.02, fontsize=11)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _render_pyvista(genome: np.ndarray, title: str, save_path):
    plotter = pv.Plotter(off_screen=save_path is not None)

    for mat, color in MAT_COLORS.items():
        coords = np.argwhere(genome == mat)
        for x, y, z in coords:
            cube = pv.Cube(center=(x, y, z),
                           x_length=0.95, y_length=0.95, z_length=0.95)
            plotter.add_mesh(cube, color=color, opacity=0.9)

    plotter.add_text(title, font_size=12)
    plotter.show_grid()

    if save_path:
        plotter.screenshot(save_path)
    else:
        plotter.show()

    return plotter



# MATPLOTLIB BACKEND

def _draw_voxels_matplotlib(ax, genome: np.ndarray):
    """Draw coloured voxel cubes on a Matplotlib 3-D axis."""
    for mat, color in MAT_COLORS.items():
        coords = np.argwhere(genome == mat)
        for x, y, z in coords:
            _draw_cube(ax, x, y, z, color)

    X, Y, Z = genome.shape
    ax.set_xlim(0, X); ax.set_ylim(0, Y); ax.set_zlim(0, Z)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")


def _draw_cube(ax, x, y, z, color, alpha=0.7, size=0.9):
    """Draw a single coloured cube at voxel position (x,y,z)."""
    s = size / 2
    faces = [
        [[x-s,y-s,z-s],[x+s,y-s,z-s],[x+s,y+s,z-s],[x-s,y+s,z-s]],  # bottom
        [[x-s,y-s,z+s],[x+s,y-s,z+s],[x+s,y+s,z+s],[x-s,y+s,z+s]],  # top
        [[x-s,y-s,z-s],[x-s,y+s,z-s],[x-s,y+s,z+s],[x-s,y-s,z+s]],  # left
        [[x+s,y-s,z-s],[x+s,y+s,z-s],[x+s,y+s,z+s],[x+s,y-s,z+s]],  # right
        [[x-s,y-s,z-s],[x+s,y-s,z-s],[x+s,y-s,z+s],[x-s,y-s,z+s]],  # front
        [[x-s,y+s,z-s],[x+s,y+s,z-s],[x+s,y+s,z+s],[x-s,y+s,z+s]],  # back
    ]
    poly = Poly3DCollection(faces, alpha=alpha,
                             facecolor=color, edgecolor="black", linewidth=0.2)
    ax.add_collection3d(poly)


def _add_legend_matplotlib(fig):
    patches = [
        plt.Rectangle((0,0),1,1, color=MAT_COLORS[mat], label=MAT_NAMES[mat])
        for mat in [PASSIVE, ACTIVE_P, ACTIVE_N]
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.9)


def _render_matplotlib(genome: np.ndarray, title: str, save_path):
    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection="3d")
    _draw_voxels_matplotlib(ax, genome)
    ax.set_title(title)
    _add_legend_matplotlib(fig)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig
