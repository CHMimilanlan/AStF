import numpy as np
import yaml

from data.animation.InverseKinematics import JacobianInverseKinematics
from data.animation.Quaternions import Quaternions
from data.animation import BVH
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap


class SaveBVH:
    def __init__(self, args):

        preprocess = np.load(args.dist_datapath)
        self.Xmean = preprocess['Xmean']
        self.Xstd = preprocess['Xstd']

    def save_output(self, output, traj, color, filename='output'):

        output = denormalize(output, self.Xmean[:7], self.Xstd[:7])
        output = np.transpose(output, (1, 2, 0))

        traj = denormalize(traj, self.Xmean[-4:], self.Xstd[-4:])
        traj = np.transpose(traj, (1, 2, 0))

        # original output
        positions = restore_animation(output[:, :, :3], traj)
        # plot_static_3d_motion(positions, color_by='frame', save_path='3d_motion_by_frame.png')

        num = 6
        frames = positions.shape[0]
        downfact = frames // num
        positions = positions[::downfact]
        frames = positions.shape[0]
        plot3d(positions[:frames//2], f"{filename}_0", color)
        plot3d(positions[frames//2:], f"{filename}_1", color)

        print('Saving animation of %s in bvh...' % filename)
        # to_bvh_cmu(positions, filename=filename, frametime=1.0/30.0)


    def save_split1(self, output, traj, color, filename='output'):
        output = denormalize(output, self.Xmean[:7], self.Xstd[:7])
        output = np.transpose(output, (1, 2, 0))

        traj = denormalize(traj, self.Xmean[-4:], self.Xstd[-4:])
        traj = np.transpose(traj, (1, 2, 0))
        # original output
        positions = restore_animation(output[:, :, :3], traj)
        # plot_static_3d_motion(positions, color_by='frame', save_path='3d_motion_by_frame.png')
        num = 6
        frames = positions.shape[0]
        downfact = frames // num
        positions = positions[::downfact]
        frames = positions.shape[0]
        plot3d_split(positions[:frames//2+1], f"{filename}_0", color)
        plot3d_split(positions[frames//2 -1:], f"{filename}_1", color)
        print('Saving animation of %s in bvh...' % filename)



def rotate_x_90(data):
    rotation_matrix = np.array([[1, 0, 0],
                                [0, 0, -1],
                                [0, 1, 0]])
    return data @ rotation_matrix.T

def plot3d(datasets, filename, color):
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (0, 9), (9, 10), (10, 11), (11, 12), (11, 13), (13, 14), (14, 15), (15, 16),
             (11, 17), (17, 18), (18, 19), (19, 20)]  

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    trans = 20
    for idx, data in enumerate(datasets):
        rotated_data = rotate_x_90(data)
        ax.scatter(rotated_data[:, 0], rotated_data[:, 1] - idx * trans, rotated_data[:, 2], s=5, c=color)

        for start, end in pairs:
            ax.plot([rotated_data[start, 0], rotated_data[end, 0]],
                    [rotated_data[start, 1] - idx * trans, rotated_data[end, 1] - idx * trans],
                    [rotated_data[start, 2], rotated_data[end, 2]], color=color, linewidth=3)

    ax.set_axis_off()
    ax.set_ylim(-len(datasets) * trans, trans)

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')

    # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 3, 1, 1]))

    ax.view_init(elev=0, azim=-15)
    # plt.show()    
    fig.savefig(f"{filename}.png", transparent=True)
    # fig.savefig(f"{filename}.jpg")

def plot3d_split(datasets, filename, color):
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
             (0, 9), (9, 10), (10, 11), (11, 12), (11, 13), (13, 14), (14, 15), (15, 16),
             (11, 17), (17, 18), (18, 19), (19, 20)]  

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    trans = 20
    for idx, data in enumerate(datasets):
        rotated_data = rotate_x_90(data)
        ax.scatter(rotated_data[:, 0], rotated_data[:, 1] - idx * trans, rotated_data[:, 2], s=5, c=color)

        for start, end in pairs:
            ax.plot([rotated_data[start, 0], rotated_data[end, 0]],
                    [rotated_data[start, 1] - idx * trans, rotated_data[end, 1] - idx * trans],
                    [rotated_data[start, 2], rotated_data[end, 2]], color=color, linewidth=3, linestyle='--')

    ax.set_axis_off()
    ax.set_ylim(-len(datasets) * trans, trans)

    ax.view_init(elev=0, azim=-15)
    fig.savefig(f"{filename}.png", transparent=True)



def my_save_output(output, traj, filename='output.bvh'):
    # output = denormalize(output, self.Xmean[:7], self.Xstd[:7])
    output = np.transpose(output, (1, 2, 0))
    # traj = denormalize(traj, self.Xmean[-4:], self.Xstd[-4:])
    traj = np.transpose(traj, (1, 2, 0))
    # original output
    positions = restore_animation(output[:, :, :3], traj)
    print('Saving animation of %s in bvh...' % filename)
    to_bvh_cmu(positions, filename=filename, frametime=1.0/30.0)



def restore_animation(pos, traj, start=None, end=None):
    """
    :param pos: (F, J, 3)
    :param traj: (F, J, 4)
    :param start: start frame index
    :param end: end frame index
    :return: positions
    """
    if start is None:
        start = 0
    if end is None:
        end = len(pos)

    Rx = traj[start:end, 0, -4]
    Ry = traj[start:end, 0, -3]
    Rz = traj[start:end, 0, -2]
    Rr = traj[start:end, 0, -1]

    rotation = Quaternions.id(1)
    translation = np.array([[0, 0, 0]])

    for fi in range(len(pos)):
        pos[fi, :, :] = rotation * pos[fi]
        pos[fi] = pos[fi] + translation[0]  # NOTE: xyz-translation
        rotation = Quaternions.from_angle_axis(-Rr[fi], np.array([0, 1, 0])) * rotation
        translation = translation + rotation * np.array([Rx[fi], Ry[fi], Rz[fi]])
    global_positions = pos

    return global_positions







def to_bvh_cmu(targets, filename, silent=True, frametime=1.0/60.0):
    """
    from 21 to 31 joints
    """
    rest, names, _ = BVH.load('data/rest_cmu.bvh')
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)

    sdr_l, sdr_r, hip_l, hip_r = 13, 17, 1, 5
    across1 = targets[:, hip_l] - targets[:, hip_r]
    across0 = targets[:, sdr_l] - targets[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[...,np.newaxis]

    forward = np.cross(across, np.array([[0,1,0]]))
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

    anim.positions[:,0] = targets[:,0]
    anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]

    mapping = {
        0: 0,
        2: 1, 3: 2, 4: 3, 5: 4,
        7: 5, 8: 6, 9: 7, 10: 8,
        12: 9, 13: 10, 15: 11, 16: 12,
        18: 13, 19: 14, 20: 15, 22: 16,
        25: 17, 26: 18, 27: 19, 29: 20,
    }

    targetmap = {}
    for k in mapping:
        targetmap[k] = targets[:, mapping[k]]

    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=silent)
    ik()
    plot3d(anim.positions[0])
    BVH.save(filename, anim, names, frametime=frametime)


def denormalize(x, mean, std):
        x = x * std + mean
        return x
