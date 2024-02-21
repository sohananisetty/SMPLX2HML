import io
from textwrap import wrap

import imageio
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

t2m_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

smplx_kinematic_chain = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

smplx_left_hand_chain = [
    [20, 28, 29, 30],
    [20, 31, 32, 33],
    [20, 25, 26, 27],
    [20, 22, 23, 24],
    [20, 34, 35, 36],
]

smplx_right_hand_chain = [
    [21, 43, 44, 45],
    [21, 46, 47, 48],
    [21, 40, 41, 42],
    [21, 37, 38, 39],
    [21, 49, 50, 51],
]


smplx_only_hand_chain = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
]
smplx_only_hand_chain2 = [
    [15, 16, 17],
    [18, 19, 20],
    [21, 22, 23],
    [24, 25, 26],
    [27, 28, 29],
]

smplx_only_hand_chain3 = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
]


smplx_full_kinematic_chain = (
    smplx_kinematic_chain + smplx_left_hand_chain + smplx_right_hand_chain
)


def plot_3d_motion(joints, title=None):
    matplotlib.use("Agg")

    data = joints.copy().reshape(len(joints), -1, 3)
    print(data.shape)

    nb_joints = joints.shape[1]
    if nb_joints == 52:
        smpl_kinetic_chain = smplx_full_kinematic_chain
    elif nb_joints == 30:
        smpl_kinetic_chain = smplx_only_hand_chain + smplx_only_hand_chain2
    elif nb_joints == 15:
        smpl_kinetic_chain = smplx_only_hand_chain
    elif nb_joints == 10:
        smpl_kinetic_chain = smplx_only_hand_chain3
    else:
        smpl_kinetic_chain = t2m_kinematic_chain

    limits = 0.5 if nb_joints == 15 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = [
        "red",
        "blue",
        "black",
        "red",
        "blue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkblue",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
        "darkred",
    ]
    frame_number = data.shape[0]
    #     print(data.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz],
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = (
            plt.figure(figsize=(480 / 96.0, 320 / 96.0), dpi=96)
            if nb_joints == 21
            else plt.figure(figsize=(10, 10), dpi=96)
        )
        if title is not None:
            wraped_title = "\n".join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        init()

        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(
            MINS[0] - trajec[index, 0],
            MAXS[0] - trajec[index, 0],
            0,
            MINS[2] - trajec[index, 1],
            MAXS[2] - trajec[index, 1],
        )
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(
                trajec[:index, 0] - trajec[index, 0],
                np.zeros_like(trajec[:index, 0]),
                trajec[:index, 1] - trajec[index, 1],
                linewidth=1.0,
                color="blue",
            )
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(
                data[index, chain, 0],
                data[index, chain, 1],
                data[index, chain, 2],
                linewidth=linewidth,
                color=color,
            )
        #         print(trajec[:index, 0].shape)

        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        io_buf = io.BytesIO()
        fig.savefig(io_buf, format="raw", dpi=96)
        io_buf.seek(0)
        # print(fig.bbox.bounds)
        arr = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
        io_buf.close()
        plt.close()
        return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None, fps=20):
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        out.append(
            plot_3d_motion(
                smpl_joints_batch[i],
                title_batch[i] if title_batch is not None else None,
            )
        )
        if outname is not None:
            imageio.mimsave(
                outname[i], np.array(out[-1]), duration=int(1000 * (1 / fps))
            )
    out = torch.stack(out, axis=0)
    return out


def render(smpl_joints, outname=None, fps=30):
    out = plot_3d_motion(
        smpl_joints,
    )

    if outname is not None:
        imageio.mimsave(outname, np.array(out), duration=int(1000 * (1 / fps)))
