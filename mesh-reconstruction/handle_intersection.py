import numpy as np
import matplotlib.pyplot as plt

class Island:
    def __init__(self, g, id):
        self.ROW = g.shape[0]
        self.COL = g.shape[1]
        self.graph = g
        self.target = id
        self.visited = [[False for j in range(self.COL)] for i in range(self.ROW)]

    def isSafe(self, i, j):
        return (i >= 0 and i < self.ROW and j >= 0 and j < self.COL)

    def DFS(self, i, j, mask):
        rowNbr = [-1, 0, 0, 1, 1, 1, -1, -1]
        colNbr = [0, -1, 1, 0, 1, -1, 1, -1]

        stack = [(i, j)]

        while stack:
            curr_i, curr_j = stack.pop()

            if not self.isSafe(curr_i, curr_j) or self.visited[curr_i][curr_j]:
                continue

            self.visited[curr_i][curr_j] = True

            if self.graph[curr_i][curr_j] == self.target:
                mask[curr_i][curr_j] = 1
            else:
                continue

            for k in range(8):
                stack.append((curr_i + rowNbr[k], curr_j + colNbr[k]))
        
        return mask

def check_safe(x, y):

    if x >=0 and x <= 1079 and y >= 0 and y <= 1919:
        return True
    return False
    
def find_depth(index_x, index_y, rest_region, depth):
    
    x_list = [-1, 0, 1, -1, 1, -1, 0, 1]
    y_list = [-1, -1, -1, 0, 0, 1, 1, 1]
    depth_list = []

    for i in range(len(x_list)):
        if check_safe(index_x + x_list[i], index_y + y_list[i]) == True:
            if rest_region[index_x + x_list[i]][index_y + y_list[i]] == 1:
                depth_list.append(depth[index_x + x_list[i]][index_y + y_list[i]])

    return sum(depth_list) / len(depth_list)
        
        

    
def BFS_depth(depth, rest_region, intersection):

    # find neighbour of intersection and uncertain region
    # By moving rest region up/down/left/right
    check_map = np.zeros(depth.shape)
    x_list = [-1, 0, 1, -1, 1, -1, 0, 1]
    y_list = [-1, -1, -1, 0, 0, 1, 1, 1]
    shifted_up = np.roll(rest_region, shift=1, axis=0)
    shifted_down = np.roll(rest_region, shift=-1, axis=0)
    shifted_right = np.roll(rest_region, shift=1, axis=1)
    shifted_left = np.roll(rest_region, shift=-1, axis=1)
    neighbour_region = np.logical_and(np.logical_or(np.logical_or(np.logical_or(shifted_down, shifted_up), shifted_right), shifted_left), intersection)

    starter_index = np.where(neighbour_region == 1)
    queue = []
    for i in range(len(starter_index[0])):
        queue.append([starter_index[0][i], starter_index[1][i]])
        check_map[starter_index[0][i]][starter_index[1][i]] = 1
    while queue:

        current_pixel = queue.pop(0)
        index_x = current_pixel[0]
        index_y = current_pixel[1]

        
        depth[index_x][index_y] = find_depth(index_x, index_y, rest_region, depth)
        intersection[index_x][index_y] = 0
        rest_region[index_x][index_y] = 1

        # TODO: add new element to queue
        for i in range(len(x_list)):

            if check_safe(index_x+x_list[i], index_y+y_list[i]) == True:
                if check_map[index_x+x_list[i]][index_y+y_list[i]] == 1:
                    continue
                if intersection[index_x+x_list[i]][index_y+y_list[i]] == 1:
                    queue.append([index_x+x_list[i], index_y+y_list[i]])
                    check_map[index_x+x_list[i]][index_y+y_list[i]] = 1
    return depth


class Handle_intersection:

    def __init__(self, depth, uncertain, semantic):
        self.depth = depth
        self.uncertain = uncertain
        self.semantic = semantic
        self.visited = np.zeros(depth.shape)
        self.foreground_id_list = [17, 19, 20, 24, 25, 26, 27, 28, 31, 32, 33]

    def handle_intersection(self):

        for i in range(self.uncertain.shape[0]):
            for j in range(self.uncertain.shape[1]):
                if self.uncertain[i][j] == 0 or self.semantic[i][j] not in self.foreground_id_list or self.visited[i][j] == 1:
                    self.visited[i][j] = 1
                    continue
                
                # If found foreground object, find its mask
                current_id = self.semantic[i][j]
                island = Island(self.semantic, current_id)
                current_mask = np.zeros(self.depth.shape)
                current_mask = island.DFS(i, j, current_mask)

                # find the intersection region
                intersection = np.logical_and(current_mask, self.uncertain)
                
                # find rest of region
                rest_region = current_mask - intersection
                if rest_region.sum() == 0:
                    self.visited[intersection == 1] = 1
                    self.visited[current_mask == 1] = 1
                    continue

                # Update depth at intersection region to average of rest region
                # avg_depth = np.sum(self.depth * rest_region) / np.sum(rest_region)
                # self.depth[intersection == 1] = avg_depth

                # Use a BFS method to update depth a intersection region
                BFS_depth(self.depth, rest_region, intersection)

                self.visited[intersection == 1] = 1
                self.visited[current_mask == 1] = 1
        
        return self.depth

