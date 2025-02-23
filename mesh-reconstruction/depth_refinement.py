import calibration
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import sys


def draw_normal(normal, out_path):
    norms = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= norms
    img_normal = (255 * 0.5*(normal+1)).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(out_path, img_normal)


class DepthRefine:

    def __init__(self, args, depth, normals, uncertain, semantics):
        self.K = calibration.__dict__[args['calibration']]

        self.lr = args['depth_refine_lr']
        self.iterations = args['depth_refine_iter']
        self.device = torch.device('cuda')

        self.depth = torch.tensor(depth, dtype=torch.float32, requires_grad=True, device=self.device)
        self.depth_ref = torch.tensor(depth, dtype=torch.float32, requires_grad=False, device=self.device)
        self.normal = torch.tensor(normals, dtype=torch.float32, requires_grad=False, device=self.device)

        self.optimizer = optim.Adam([self.depth], lr=self.lr)

        self.l1 = args['depth_refine_l1']
        self.l2 = args['depth_refine_l2']
        self.l3 = args['depth_refine_l3']

        self.semantics = semantics
        self.weight = torch.tensor(self.get_weight()).cuda()
        self.uncertain_map = torch.tensor(1 - uncertain, dtype=torch.float32, requires_grad=False, device="cuda") * self.weight
        self.normal_new = self.normal * self.weight.unsqueeze(-1)

    def get_weight(self):

        weight = np.ones(self.depth.shape) * 5

        # reduce weight for roads and side walks
        weight[self.semantics == 7] = 0.0001
        weight[self.semantics == 8] = 0.0001

        return weight

    def get_depth_partials(self):

        # u, v mean the pixel coordinate in the image
        dz_dv, dz_du = torch.gradient(self.depth)

        # u*depth = fx*x + cx --> du/dx = fx / depth
        # BUG: what if depth == 0 ?
        du_dx = self.K['fx'] / self.depth  # x is xyz of camera coordinate
        dv_dy = self.K['fy'] / self.depth

        dz_dx = dz_du * du_dx
        dz_dy = dz_dv * dv_dy

        return dz_dx, dz_dy

    def get_surface_normal_by_depth(self):

        dz_dx, dz_dy = self.get_depth_partials()

        # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
        normal_cross = torch.dstack( (-dz_dx, -dz_dy, torch.ones_like(self.depth)))
        # normalize to unit vector
        normal_unit = normal_cross / torch.norm(normal_cross, dim=-1, keepdim=True)
        # set default normal to [0, 0, 1]
        # normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
        return normal_unit

    def get_gradient_loss(self):

        dz_dx, dz_dy = self.get_depth_partials()

        vec_x = torch.dstack((torch.ones_like(self.depth), torch.zeros_like(self.depth), dz_dx))
        vec_y = torch.dstack((torch.zeros_like(self.depth), torch.ones_like(self.depth), dz_dy))
        # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
        dot_x = torch.einsum('ijk, ijk->ij', vec_x, self.normal)
        dot_y = torch.einsum('ijk, ijk->ij', vec_y, self.normal)

        return torch.mean(dot_x * dot_x * self.uncertain_map) + torch.mean(dot_y * dot_y * self.uncertain_map)

    def optimize(self):

        pbar = tqdm(total=self.iterations, desc="Optimizing Depth", dynamic_ncols=True)
        for i in range(self.iterations):

            normal_pred = self.get_surface_normal_by_depth()

            norm_loss = F.mse_loss( normal_pred * self.weight.unsqueeze(-1), self.normal_new)
            cont_loss = self.get_gradient_loss()
            depth_loss = F.mse_loss(self.depth, self.depth_ref)

            loss = self.l1 * norm_loss + self.l2 * cont_loss + self.l3 * depth_loss
            pbar.update(1)
            pbar.set_description(f"Loss: {loss:.4f}")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return self.depth.detach().cpu().numpy()

