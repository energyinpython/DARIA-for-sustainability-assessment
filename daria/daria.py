import numpy as np


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
            if np.mean(R[:, i]) != 0:
                for p in range(0, t):
                    for k in range(0, t):
                        Yi[p] += np.abs(R[p, i] - R[k, i]) / (2 * t**2 * (np.sum(R[:, i]) / t))
            else:
                for p in range(0, t):
                    for k in range(0, t):
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
            if np.mean(DM[:, j]) != 0:
                for p in range(0, t):
                    for k in range(0, t):
                        Yj[p] += np.abs(DM[p, j] - DM[k, j]) / (2 * t**2 * (np.sum(DM[:, j]) / t))
            else:
                for p in range(0, t):
                    for k in range(0, t):
                        Yj[p] += np.abs(DM[p, j] - DM[k, j]) / (t**2 - t)
            G[j] = np.sum(Yj)
        return G

    
    def _direction(self, R, descending):
        t, m = np.shape(R)
        direction_list = []
        dir_class = []
        # iteration over alternatives i=1, 2, ..., m
        for i in range(m):
            thresh = 0
            # iteration over periods p=1, 2, ..., t
            for p in range(1, t):
                thresh += R[p, i] - R[p - 1, i]
            # there are rankings so crits are cost type
            if descending == False:
                if thresh < 0:
                    direction_list.append(r'$\uparrow$')
                    dir_class.append(1)
                elif thresh > 0:
                    direction_list.append(r'$\downarrow$')
                    dir_class.append(-1)
                else:
                    direction_list.append(r'$=$')
                    dir_class.append(0)
            elif descending == True:
                if thresh < 0:
                    direction_list.append(r'$\downarrow$')
                    dir_class.append(-1)
                elif thresh > 0:
                    direction_list.append(r'$\uparrow$')
                    dir_class.append(1)
                else:
                    direction_list.append(r'$=$')
                    dir_class.append(0)
        return direction_list, dir_class


    def _direction_criteria(self, DM, crit_types):
        t, n = np.shape(DM)
        direction_list = []
        # iteration over criteria j=1, 2, ..., n
        for j in range(n):
            thresh = 0
            # iteration over periods p=1, 2, ..., t
            for p in range(1, t):
                thresh += DM[p, j] - DM[p - 1, j]
            # if criterion is cost type
            if crit_types[j] == -1:
                if thresh < 0:
                    direction_list.append(r' $\uparrow$')
                elif thresh > 0:
                    direction_list.append(r' $\downarrow$')
                else:
                    direction_list.append(r' $=$')
            # if criterion is profit type
            else:
                if thresh > 0:
                    direction_list.append(r' $\uparrow$')
                elif thresh < 0:
                    direction_list.append(r' $\downarrow$')
                else:
                    direction_list.append(r' $=$')
        return direction_list


    def _update_efficiency(self, S, G, dir, descending):
        if descending == False:
            S = 1 - S
        final_S = S + G * dir
        if descending == False:
            final_S = 1 - final_S
        return final_S