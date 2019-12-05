import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def detrend(coordinates, pca_data, order=1, verbose=False, plot_points=False,
            goodness_of_fit=True):
    """https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6"""

    x = coordinates["X"].values
    y = coordinates["Y"].values
    z = pca_data.values
    data = np.c_[x,y,z]

    # regular grid covering the domain of the data
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20),
                      np.linspace(mn[1], mx[1], 20))
    XX = X.flatten()
    YY = Y.flatten()

    # 1: linear
    if order == 1:
        # best-fit linear plane
        A = np.c_[data[:,0],
                  data[:,1],
                  np.ones(data.shape[0])]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients

        Z_ = C[0]*x + C[1]*y + C[2]

        # evaluate it on grid
        Z = C[0]*X + C[1]*Y + C[2]

        # or expressed using matrix/vector product
        #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

        if verbose:
            print("solution:")
            print(f"Z = {C[0]}x + {C[1]} y + {C[2]}")

    # 2: quadratic
    elif order == 2:
        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]),
                  data[:,:2],
                  np.prod(data[:,:2], axis=1),
                  data[:,:2]**2]
        C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])  # coefficients

        Z_ = C[4]*x**2. + C[5]*y**2. + C[3]*x*y + C[1]*x + C[2]*y + C[0]

        # evaluate it on a grid
        Z = C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]

        # or expressed using matrix/vector product
        # Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

        if verbose:
            print("solution:")
            print(f"""Z = {C[4]}x² + {C[5]}y² + {C[3]}x*y + {C[1]}x + {C[2]}y
                   + {C[0]}""")


    # plot points and fitted surface
    if plot_points:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        # ax.axis('equal')
        # ax.axis('tight')
        plt.show();

    if goodness_of_fit:
        # https://stackoverflow.com/questions/29003241/how-to-quantitatively-measure-goodness-of-fit-in-scipy
        # residual sum of squares
        ss_res = np.sum((z - Z_) ** 2)

        # total sum of squares
        ss_tot = np.sum((z - np.mean(z)) ** 2)

        # r-squared
        r2 = 1 - (ss_res / ss_tot)

    errors = np.squeeze(np.asarray(z - Z_))

    return errors, r2
