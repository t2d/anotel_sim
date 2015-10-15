import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import csv
import sys


def main(filename='sim3d.csv'):

    X = []
    Y = []
    Z = []

    # read data
    with open(filename, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            X.append(int(row[0]))
            Y.append(int(row[1]))
            Z.append(float(row[2]))
        f.close()

    # make arrays
    xi = len(set(X))
    yi = len(set(Y))
    # zi = len(set(Z))
    XX = np.reshape(X, (xi, yi))
    YY = np.reshape(Y, (xi, yi))
    ZZ = np.reshape(Z, XX.shape)

    # plot
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_color_cycle(['k'])
    ax.plot_wireframe(XX, YY, ZZ, color="black")
    # ax.plot_surface(XX, YY, ZZ)
    # labels
    ax.set_xlabel('Users')
    ax.set_ylabel('Events')
    ax.set_zlabel('Matched paths')
    # ax.set_xlim(0, X[-1])
    # ax.set_ylim(X[-1], 0)
    # ax.set_zlim(0, 100)

    plt.show()
    plt.savefig('sim3d.svg', format='svg', bbox_inches='tight')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
