import numpy as np
import copy
import itertools

class DARIA():
    def __init__(self):
        pass

    def _gini(self, R):
        t, m = np.shape(R)
        G = np.zeros(m)
        # iteration over alternatives i=1, 2, ..., m
        for i in range(0, m):
            # iteration over periods p=1, 2, ..., t
            Yi = np.zeros(t)
            if np.mean(R[:, i]):
                for p, k in itertools.product(range(t), range(t)):
                    Yi[p] += np.abs(R[p, i] - R[k, i]) / (2 * t**2 * (np.sum(R[:, i]) / t))
            else:
                for p, k in itertools.product(range(t), range(t)):
                    Yi[p] += np.abs(R[p, i] - R[k, i]) / (t**2 - t)
            G[i] = np.sum(Yi)
        return G


    def _gini_criteria(self, DM):
        t, n = np.shape(DM)
        G = np.zeros(n)
        # iteration over criteria j=1, 2, ..., n
        for j in range(0, n):
            Yj = np.zeros(t)
            # iteration over periods p=1, 2, ..., t
            if np.mean(DM[:, j]):
                for p, k in itertools.product(range(t), range(t)):
                    Yj[p] += np.abs(DM[p, j] - DM[k, j]) / (2 * t**2 * (np.sum(DM[:, j]) / t))
            else:
                for p, k in itertools.product(range(t), range(t)):
                    Yj[p] += np.abs(DM[p, j] - DM[k, j]) / (t**2 - t)
            G[j] = np.sum(Yj)
        return G


    # for MCDA methods type = 1: descending order: higher is better, type -1: opposite
    def _direction(self, R, type):
        t, m = np.shape(R)
        direction_list = []
        dir_class = np.zeros(m)
        # iteration over alternatives i=1, 2, ..., m
        for i in range(m):
            thresh = 0
            # iteration over periods p=1, 2, ..., t
            for p in range(1, t):
                thresh += R[p, i] - R[p - 1, i]
            # classification based on thresh
            dir_class[i] = np.sign(thresh)
           
        direction_array = copy.deepcopy(dir_class)
        direction_array = direction_array * type
        for i in range(len(direction_array)):
            if direction_array[i] == 1:
                direction_list.append(r'$\uparrow$')
            elif direction_array[i] == -1:
                direction_list.append(r'$\downarrow$')
            elif direction_array[i] == 0:
                direction_list.append(r'$=$')
        return direction_list, dir_class


    def _direction_criteria(self, DM, crit_types):
        t, n = np.shape(DM)
        direction_list = []
        dir_class = np.zeros(n)
        # iteration over criteria j=1, 2, ..., n
        for j in range(n):
            thresh = 0
            # iteration over periods p=1, 2, ..., t
            for p in range(1, t):
                thresh += DM[p, j] - DM[p - 1, j]
            
            dir_class[j] = np.sign(thresh)

        direction_array = copy.deepcopy(dir_class)
        direction_array = direction_array * crit_types
        for i in range(len(direction_array)):
            if direction_array[i] == 1:
                direction_list.append(r'$\uparrow$')
            elif direction_array[i] == -1:
                direction_list.append(r'$\downarrow$')
            elif direction_array[i] == 0:
                direction_list.append(r'$=$')
        return direction_list


    def _update_efficiency(self, S, G, dir):
        return S + G * dir