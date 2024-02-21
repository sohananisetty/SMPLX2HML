import os

import numpy as np
from trimesh import Trimesh

os.environ["PYOPENGL_PLATFORM"] = "egl"

from typing import Optional

import imageio
import pyrender
import torch
import trimesh
from core.models.smpl.smpl import SMPL
from pyrender.constants import RenderFlags
from shapely import geometry
import rotation_conversions
import smplx
from human_body_prior.body_model.body_model import BodyModel
from PIL import Image


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(
        self,
        scale,
        translation,
        znear=pyrender.camera.DEFAULT_Z_NEAR,
        zfar=None,
        name=None,
    ):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, device, model="smpl") -> None:
        self.device = device
        self.model_name = model

        if model == "smpl":
            self.smpl_model = SMPL().eval().to(device)
            self.faces = torch.Tensor(self.smpl_model.faces)

        if model == "smplx":
            male_bm_path = "/srv/hays-lab/scratch/sanisetty3/music_motion/motion-diffusion-model/body_models/smplx/SMPLX_MALE.npz"

            female_bm_path = "/srv/hays-lab/scratch/sanisetty3/music_motion/motion-diffusion-model/body_models/smplx/SMPLX_FEMALE.npz"
            male_bm = BodyModel(bm_fname=male_bm_path).to(self.device).eval()
            female_bm = BodyModel(bm_fname=female_bm_path).to(self.device).eval()
            self.faces = male_bm.f
            self.smpl_model = male_bm
            # (
            #     smplx.SMPLX(
            #         "/srv/hays-lab/scratch/sanisetty3/music_motion/TGM3D/body_models/smplx/SMPLX_NEUTRAL.npz"
            #     )
            #     .eval()
            #     .to(device)
            # )
        base_color = (0.11, 0.53, 0.8, 0.5)
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7, alphaMode="OPAQUE", baseColorFactor=base_color
        )

        self.camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
        self.light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=300)

    def process6D(self, aist_vec: torch.Tensor):
        aist_ex = aist_vec
        trans = aist_ex[:, :3]
        if aist_vec.shape[-1] == 135:
            rots = torch.cat((aist_ex[:, 3:], aist_ex[:, -12:]), 1)
        else:
            rots = aist_ex[:, 3:]
        aist_ex_9d = rotation_conversions.rotation_6d_to_matrix(
            rots.reshape(-1, 6)
        ).reshape(aist_vec.shape[0], 24, 3, 3)
        global_orient = aist_ex_9d[:, 0:1]
        rotations = aist_ex_9d[:, 1:]
        out_aist_og = self.smpl_model(
            body_pose=rotations, global_orient=global_orient, transl=trans
        )

        return out_aist_og

    def full_pose_to_smplx(self, full_pose, trans):
        assert full_pose.shape[-1] == 165
        smplx = {}
        smplx["trans"] = torch.Tensor(trans).to(self.device)
        smplx["root_orient"] = torch.Tensor(full_pose[:, 0:3]).to(self.device)
        smplx["pose_body"] = torch.Tensor(full_pose[:, 3:66]).to(self.device)
        smplx["pose_hand"] = torch.Tensor(full_pose[:, 75:165]).to(self.device)
        smplx["pose_jaw"] = torch.Tensor(full_pose[:, 66:69]).to(self.device)
        smplx["pose_eye"] = torch.Tensor(full_pose[:, 69:75]).to(self.device)

        smplx_output = self.smpl_model.forward(**smplx)

        return smplx_output

    def render(
        self,
        motion_vec: torch.Tensor,
        name: str,
        step: Optional[int] = None,
        outdir: str = "test_vis",
        fps=20,
    ):
        if self.model_name == "smpl":
            smpl_dict = self.process6D(motion_vec[:, :135].to(self.device))
            vertices = smpl_dict["vertices"].detach()

        elif self.model_name == "smplx":
            smpl_dict = self.full_pose_to_smplx(motion_vec[:, 3:], motion_vec[:, :3])
            vertices = smpl_dict.v.detach()

        print(vertices.shape)
        frames = vertices.shape[0]
        vertices = vertices - torch.Tensor([0, torch.mean(vertices[0], 0)[1], 0]).to(
            self.device
        )

        MINS = torch.min(torch.min(vertices, axis=0)[0], axis=0)[0]
        MAXS = torch.max(torch.max(vertices, axis=0)[0], axis=0)[0]

        minx = MINS[0] - 0.5
        maxx = MAXS[0] + 0.5
        minz = MINS[2] - 0.5
        maxz = MAXS[2] + 0.5
        polygon = geometry.Polygon(
            [[minx, minz], [minx, maxz], [maxx, maxz], [maxx, minz]]
        )
        polygon_mesh = trimesh.creation.extrude_polygon(polygon, 1e-5)

        bg_color = [1, 1, 1, 0.8]

        vid = []
        for i in range(frames):
            mesh = Trimesh(
                vertices=vertices[i].cpu().squeeze().tolist(),
                faces=np.array(self.faces.cpu()),
            )

            mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material)

            polygon_mesh.visual.face_colors = [0, 0, 0, 0.21]
            polygon_render = pyrender.Mesh.from_trimesh(polygon_mesh, smooth=False)
            scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

            scene.add(mesh)

            c = np.pi / 2

            scene.add(
                polygon_render,
                pose=np.array(
                    [
                        [1, 0, 0, 0],
                        [0, np.cos(c), -np.sin(c), MINS[1].cpu().numpy()],
                        [0, np.sin(c), np.cos(c), 0],
                        [0, 0, 0, 1],
                    ]
                ),
            )

            light_pose = np.eye(4)
            light_pose[:3, 3] = [0, -1, 1]
            scene.add(self.light, pose=light_pose.copy())

            light_pose[:3, 3] = [0, 1, 1]
            scene.add(self.light, pose=light_pose.copy())

            light_pose[:3, 3] = [1, 1, 2]
            scene.add(self.light, pose=light_pose.copy())

            c = -np.pi / 6

            scene.add(
                self.camera,
                pose=[
                    [1, 0, 0, (minx + maxx).cpu().numpy() / 2],
                    [0, np.cos(c), -np.sin(c), 1.5],
                    [
                        0,
                        np.sin(c),
                        np.cos(c),
                        max(
                            4,
                            minz.cpu().numpy() + (1.5 - MINS[1].cpu().numpy()) * 2,
                            (maxx - minx).cpu().numpy(),
                        ),
                    ],
                    [0, 0, 0, 1],
                ],
            )

            # render scene
            r = pyrender.OffscreenRenderer(960, 960)

            color, _ = r.render(scene, flags=RenderFlags.RGBA)
            # Image.fromarray(color).save(outdir+name+'_'+str(i)+'.png')

            vid.append(color)

            r.delete()

        out = np.stack(vid, axis=0)

        out_path = (
            os.path.join(outdir, str(step), str(name) + ".gif")
            if step is not None
            else os.path.join(outdir, str(name) + ".gif")
        )

        # ims = [Image.fromarray(a_frame) for a_frame in out]
        # ims[0].save(
        #     out_path, save_all=True, append_images=ims[1:], duration=(1000 / fps)
        # )
        imageio.mimsave(out_path, out, duration=(1000 / fps))
