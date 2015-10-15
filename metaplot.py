import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import csv
import sys


def metaplot(name='metasim', x='Users', y='Percent matched'):
    # read csv
    f = open(name + '.csv', 'rb')
    reader = csv.reader(f)

    steps = []
    mean_two = []
    conf_two = []
    mean_three = []
    conf_three = []
    mean_paths = []
    conf_paths = []

    # collect data
    for row in reader:
        users, mean_2, conf_2, mean_3, conf_3, mean_path, conf_path = row
        steps.append(int(users))
        mean_two.append(float(mean_2)/100)
        conf_two.append(float(conf_2)/100)
        mean_three.append(float(mean_3)/100)
        conf_three.append(float(conf_3)/100)
        mean_paths.append(float(mean_path)/100)
        conf_paths.append(float(conf_path)/100)

    # plot
    fig = plt.figure()
    ax = fig.gca()
    ax.set_color_cycle(['k'])
    plt.plot(steps, mean_two)
    plt.plot(steps, mean_three)
    plt.plot(steps, mean_paths)
    plt.errorbar(steps, mean_two, conf_two, label="2-Point combinations", fmt='.')
    plt.errorbar(steps, mean_three, conf_three, label="3-Point combinations", fmt='s')
    plt.errorbar(steps, mean_paths, conf_paths, label="Complete paths", fmt='^')

    # legend
    plt.rcParams['legend.loc'] = 'best'
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, numpoints=1)

    plt.xlim(0, steps[-1])
    plt.xlabel(x)
    plt.ylim(0, 1)
    plt.ylabel(y)
    plt.savefig(name + '.svg', format='svg', bbox_inches='tight')

if __name__ == "__main__":
    metaplot(sys.argv[1], sys.argv[2], sys.argv[3])
