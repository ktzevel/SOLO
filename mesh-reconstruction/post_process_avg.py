import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
from uncertain import Island
from collections import deque
from tqdm import tqdm


def create_index(shape: tuple):
    H, W = shape
    d = {}
    for i in range(H):
        for j in range(W):
            d[(i, j)] = i * 1920 + j
    return d


class PostProcessAVG:

    def __init__(self, args, scene_mesh, uncertain, semantics):

        self.vertices = scene_mesh.vertices
        self.faces = scene_mesh.faces
        self.uncertain_map = uncertain
        # self.id = (plt.imread(seg_path) * 255).astype(int)
        self.id = semantics
        
        # f = open(args['post_process_index'], "rb")
        # self.index = pickle.load(f)
        self.index = create_index((1080, 1920))
        self.margin = args['post_process_margin']
        self.foreground_id_list = [100, 17, 19, 20, 24, 25, 32, 33]

    def check_edge(self, vert_1, vert_2):
        x1, y1, z1 = vert_1
        x2, y2, z2 = vert_2
        delta_xy = math.dist([x1, y1], [x2, y2])
        delta_z = math.dist([z1], [z2])

        if delta_z > 0.01:
            return False

        return delta_z / delta_xy < 20

    def check_verts(self, x1, y1, x2, y2):
        return (self.id[x1][y1] == self.id[x2][y2])

    def check_verts2(self, x, y):
        '''
        Check if vertex inside uncertain region
        '''
        return self.uncertain_map[x][y]

    def check_verts_foreground(self, x, y):
        '''
        Check if the vertex belongs to a foreground object
        '''
        vert_seg = self.id[x][y]
        return vert_seg == 17

    def remove_unexpected(self):

        # Try two different ways of removing unexpected faces
        # 1. Remove all faces that have at least one vertex in the uncertain region
        # 2. If vertices in foreground objects, skip

        keep_face = []

        #for face in tqdm(self.faces):
        for face in self.faces:

            # assumes that the mesh vertices are 1920x1080
            vert0_ij, vert1_ij, vert2_ij = np.array(np.unravel_index(face, self.id.shape)).T
            vert0_xyz, vert1_xyz, vert2_xyz = self.vertices[face]

            # if all vertices of a face correspond to the same semantic class we keep that face
            v0_id = self.id[vert0_ij[0], vert0_ij[1]]
            v1_id = self.id[vert1_ij[0], vert1_ij[1]]
            v2_id = self.id[vert2_ij[0], vert2_ij[1]]
            if v0_id == v1_id and v1_id == v2_id and v0_id in self.foreground_id_list:

                # For some reason even if I have the function that set depth of pole to the avarage of their attached sign
                # The depth change is still not sharp.
                # Temp fix
                thr = 3
                if abs(vert0_xyz[2] - vert1_xyz[2]) > thr or abs(vert0_xyz[2] - vert2_xyz[2]) > thr or abs(vert1_xyz[2] - vert2_xyz[2]) > thr:
                    continue

                keep_face.append(face)
                continue

            # if any of the vertex falls in uncertain mask, remove the face
            if self.check_verts2(*vert0_ij) == 1 or self.check_verts2(*vert1_ij) == 1 or self.check_verts2(*vert2_ij) == 1:
                continue

            keep_face.append(face)

        self.faces = keep_face

    def find_bbox(self, island):
        '''
        TODO: replace with COCO mask's function.
        Find tight bounding box of the uncertain region
        Used to find completion region 
        '''
        left, right, up, down = -1, -1, -1, -1
        # Find bounding box
        for i in range(island.shape[0]):
            if up == -1 and np.sum(island[i, :]) > 0:
                up = i
            if down == -1 and up != -1 and np.sum(island[i, :]) == 0:
                down = i-1
        for i in range(island.shape[1]):
            if left == -1 and np.sum(island[:, i]) > 0:
                left = i
            if right == -1 and left != -1 and np.sum(island[:, i]) == 0:
                right = i-1
        return max(1, left-self.margin), min(1919, right+self.margin+1), min(1, up-self.margin), min(1079, down+self.margin+1)

    def find_first_last_ones(self, arr):
        '''
        This function finds the index of first and last 1 in arr
        '''
        first_one_index = None
        last_one_index = None

        for idx, value in enumerate(arr):
            if value == 1:
                if first_one_index is None:
                    first_one_index = idx
                last_one_index = idx

        return first_one_index, last_one_index

    def insert_points_between(self, A, B, num_points):
        '''
        Linearly insert vertices between A and B
        '''
        # Calculate the step size for interpolation
        step_size = 1.0 / (num_points + 1)

        # Initialize the list to store the interpolated points
        interpolated_points = []

        for i in range(1, num_points + 1):
            # Calculate the fraction for interpolation
            fraction = step_size * i

            # Perform linear interpolation to find the point
            interpolated_point = A + fraction * (B - A)

            # Append the interpolated point to the list
            interpolated_points.append(interpolated_point)

        return interpolated_points

    def add_vertices(self, island, island_new, island_all):

        # Ids of foreground objects
        # The original island is one of the uncertain masks
        for foreground_id in self.foreground_id_list:
            island = np.logical_or(
                island, (self.id * island_new) == foreground_id)

        island_all = np.logical_or(island_all, island)
        ignored = np.zeros(island_all.shape)

        # Add vertices
        for i in range(island.shape[0]):

            if np.sum(island[i, :]) == 0:
                island_all[i, :] = 0
                continue

            # For those are not 0, find the left most and right most index
            # For those close to image edge, ignore
            first_index, last_index = self.find_first_last_ones(island[i, :])
            if first_index - 5 < 0 or last_index + 5 > 1919:
                ignored[i, first_index:last_index] = 1
                continue

            if self.id[(i, first_index-5)] == 23 or self.id[(i, first_index+5)] == 23:
                ignored[i, first_index-5:last_index+5] = 1
                island_all[i, :] = 0
                continue

            left_vertices = self.vertices[self.index[(i, first_index-5)]]
            right_vertices = self.vertices[self.index[(i, last_index+5)]]
            inter_num = last_index - first_index - 1
            if inter_num <= 0:
                continue
            points = self.insert_points_between(
                np.array(left_vertices), np.array(right_vertices), inter_num)
            # left_depth = depth_final[i][first_index-1]
            # right_depth = depth_final[i][last_index+1]
            # depth_final[i][first_index:last_index+1] = np.linspace(left_depth, right_depth, inter_num)

            for num in range(len(points)):
                self.vertices.append(points[num])
                self.index[(i, first_index+num)] = len(self.vertices) - 1

        return island_all, ignored

    def face_complete(self, island_all, ignored):

        for row in range(island_all.shape[0]):
            for col in range(island_all.shape[1]):

                if island_all[row][col] == 0 or ignored[row][col] == 1:
                    continue

                island_all[row][col] = 0
                # Add faces
                # 1 - 2 - 3
                # | \ | / |
                # 4 - 5 - 6
                # | / | \ |
                # 7 - 8 - 9
                neighbours_list = [(-1, -1, -1, 0), (-1, -1, 0, -1), (1, -1, 0, -1), (1, -1, 1, 0),
                                   (-1, 1, -1, 0), (-1, 1, 0, 1), (1, 1, 0, 1), (1, 1, 1, 0)]
                for neighbours in neighbours_list:
                    dr1, dc1, dr2, dc2 = neighbours

                    # Filter out invalid pairs
                    # TODO: filter out invalid pairs?
                    # if island_all[row+dr1][col+dc1] != 0 or island_all[row+dr2][col+dc2] != 0:
                    #     continue

                    # Get index
                    # New vertices is the last vertex, as it start with 0, should -1
                    # Decide if they are newly added vertices of old vertices, they should follow different whys of calculating index
                    if row+dr1 >= 0 and row+dr1 <= 1079 and col+dc1 >= 0 and col+dc1 <= 1919 and row+dr2 >= 0 and row+dr2 <= 1079 and col+dc2 >= 0 and col+dc2 <= 1919:
                        if ignored[row+dr1][col+dc1] == 1 or ignored[row+dr2][col+dc2] == 1:
                            continue
                        index_new = self.index[(row, col)]
                        v1_index = self.index[(row+dr1, col+dc1)]
                        v2_index = self.index[(row+dr2, col+dc2)]
                        self.faces.append([index_new, v1_index, v2_index])

    def mesh_compeletion(self):

        self.vertices = self.vertices.tolist()

        # Find complete uncertain region
        g1 = Island(self.uncertain_map, 1)
        _, islands = g1.countIslands()
        island_all = np.zeros((1080, 1920))

        num = 0
        for island in islands:

            # Find bounding box
            island_new = np.zeros((1080, 1920))
            left, right, up, down = self.find_bbox(island)
            island_new[up:down, left:right] = 1

            # Find all missing regions, which need to calculate new depth
            island_all, ignored = self.add_vertices(
                island, island_new, island_all)

            # Find starting pixel, this method use the pixel with largest depth
            self.face_complete(island_all, ignored)

        return np.array(self.vertices), np.array(self.faces)

