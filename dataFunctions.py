import numpy as np

def make_groundtruth(filename):
    data = np.loadtxt(filename, comments="#")

    tx_ty = data[:, [1,2]]

    #tx_ty_list = [np.array([tx, ty]) for tx, ty in tx_ty]

    return tx_ty