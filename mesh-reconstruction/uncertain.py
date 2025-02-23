import numpy as np
from imantics import Polygons, Mask
import matplotlib.pyplot as plt
import scipy


class Island:

    def __init__(self, g, id):
        self.ROW = g.shape[0]
        self.COL = g.shape[1]
        self.graph = g
        self.target = id
        self.visited = [[False for j in range(
            self.COL)]for i in range(self.ROW)]

    # A function to check if a given cell
    # (row, col) can be included in DFS
    def isSafe(self, i, j, visited):
        # row number is in range, column number
        # is in range and value is 1
        # and not yet visited
        return (i >= 0 and i < self.ROW and j >= 0 and j < self.COL)

    # A utility function to do DFS for a 2D
    # boolean matrix. It only considers
    # the 8 neighbours as adjacent vertices

    def DFS(self, i, j, mask):

        stack = [(i, j)]

        rowNbr = [-1, 0, 0, 1, 1, 1, -1, -1]
        colNbr = [0, -1, 1, 0, 1, -1, 1, -1]

        while stack:

            i, j = stack.pop()

            if not self.isSafe(i, j, self.visited):
                continue

            if self.visited[i][j]:
                continue

            self.visited[i][j] = True

            if self.graph[i][j] == self.target:
                mask[i][j] = 1
            else:
                continue

            # Push connected neighbours to stack
            for k in range(8):
                stack.append((i + rowNbr[k], j + colNbr[k]))

        return mask

    # The main function that returns count of islands in a given boolean 2D matrix

    def countIslands(self):

        # Initialize count as 0 and traverse through the all cells of given matrix
        count = 0
        mask_list = []
        for i in range(self.ROW):
            for j in range(self.COL):
                # If a cell with value 1 is not visited yet,
                # then new island found
                if self.visited[i][j] == False and self.graph[i][j] == self.target:
                    # Visit all cells in this island
                    # and increment island count
                    mask = np.zeros((self.ROW, self.COL))
                    mask_list.append(self.DFS(i, j, mask))
                    count += 1

        return count, mask_list


def find_connected(list1, list2):
    '''
    Find connected pairs of mask in list1 and list2
    '''
    connected_pairs = []
    for island_1 in list1:
        for island_2 in list2:
            mask = np.logical_or(island_1, island_2)
            polygon = Mask(mask).polygons()
            if len(polygon.segmentation) == 1:
                connected_pairs.append([island_1, island_2])

    return connected_pairs


def modify_depth(mask1, mask2, depth):
    '''
    Set depth of mask1 to the average depth of mask2
    '''
    sign_depth = np.sum(mask2 * depth) / np.sum(mask2)
    depth[mask1 == 1] = sign_depth

    pole_depth = np.sum(mask1 * depth) / np.sum(mask1)
    depth[mask2 == 1] = pole_depth

    return depth


def get_unique_numbers(array):

    unique_numbers = set()
    for row in array:
        for num in row:
            unique_numbers.add(num)
    return sorted(list(unique_numbers))


def check_if_in(array):
    '''
    Check if uncertain mask contains a foreground object pixel.
    Foreground objects: pole, traffic signs, traffic light, person, rider, car, 
    truck, bus, train, motorcycle, bicycle
    '''
    valid_numbers = [100, 17, 19, 20, 24, 25, 32, 33]

    # Check if all elements in array are the same
    if np.all(array == array[0][0]):
        return False

    # Check if any valid number is present in the array
    return np.any(np.isin(array, valid_numbers))


def check_if_in_all(array):
    '''
    Check if uncertain mask contains a foreground object pixel.
    Foreground objects: pole, traffic signs, traffic light, person, rider, car, 
    truck, bus, train, motorcycle, bicycle
    '''
    valid_numbers = [100, 17, 19, 20, 24, 25, 26, 27, 28, 31, 32, 33]

    # Check if all elements in array are the same
    if np.all(array == array[0][0]):
        return False

    # Check if any valid number is present in the array
    return np.any(np.isin(array, valid_numbers))


def find_uncertain_area(depth, segmentation, filter_size, limit=1e-4):
    '''
    Dual-reference variance filter
    This function takes in depth and segmentation map of a image,
    then uses a filter with size filter_size * filter_size to find 
    the untertain depth area of a predicted depth map. 
    '''
    assert np.array_equal(
        depth.shape, segmentation.shape), "depth map and segmentation map must have the same shape."
    mid = (depth.max() + depth.min()) / 2
    uncertain_map = np.zeros((depth.shape))
    uncertain_map_optimize = np.zeros((depth.shape))

    # TODO: speed this up
    for i in range(depth.shape[0]-filter_size+1):
        for j in range(depth.shape[1]-filter_size+1):

            check_area_depth = depth[i:i+filter_size, j:j+filter_size]

            # Ignore if too far
            if check_area_depth.max() > mid:
                continue

            # Check if the region contains foreground objects
            check_area_segmentation = segmentation[i:i + filter_size, j:j+filter_size]
            if not check_if_in(check_area_segmentation):
                if check_if_in_all(check_area_segmentation):
                    uncertain_map_optimize[i:i + filter_size, j:j+filter_size] = 1
                continue

            # Update the uncertain map
            if np.var(check_area_depth) > limit:
                uncertain_map[i:i+filter_size, j:j+filter_size] = 1
                uncertain_map_optimize[i:i+filter_size, j:j+filter_size] = 1

    return uncertain_map, uncertain_map_optimize


def uncertain(args, semantics, depth):
    """ accepts normalized depth """

    # Load segmentation annotations
    # data = (plt.imread(segentation_file) * 255).astype(int)
    uncertain_area, uncertain_area_opt = find_uncertain_area(
        depth, semantics, args["uncertain_filter_size"], args["uncertain_variance_limit"])

    return uncertain_area, uncertain_area_opt


def modify_pole(args, semantics, depth):
    # For pole with a sign attached modify depth of rod to be the average depth of sign
    # TODO: For pole without sign.
    # TODO: use scipy vectorized functions instead.
    g_17 = Island(semantics, 17)  # pole
    _, island_17 = g_17.countIslands()
    g_20 = Island(semantics, 20)  # sign
    _, island_20 = g_20.countIslands()
    g_0 = Island(semantics, 100)  # null signs
    _, island_0 = g_0.countIslands()
    pair1 = find_connected(island_17, island_0)
    pair2 = find_connected(island_17, island_20)

    for i in range(len(pair1)):
        # plt.clf()
        # plt.imshow(pair1[i][0] == 1)
        # plt.savefig(f"{i}.png")
        # plt.close()
        depth = modify_depth(pair1[i][0], pair1[i][1], depth)

    # exit()

    for i in range(len(pair2)):
        depth = modify_depth(pair2[i][0], pair2[i][1], depth)

    return depth
