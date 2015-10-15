# import https://stackoverflow.com/questions/7358118/matplotlib-black-white-colormap-with-dashes-dots-etc
def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    COLORMAP = {
        'b': {'marker': None, 'dash': (None, None)},
        'g': {'marker': None, 'dash': [5, 5]},
        'r': {'marker': None, 'dash': [5, 3, 1, 3]},
        'c': {'marker': None, 'dash': [1, 3]},
        'm': {'marker': None, 'dash': [5, 2, 5, 2, 5, 10]},
        'y': {'marker': None, 'dash': [5, 3, 1, 2, 1, 10]},
        'k': {'marker': 'o', 'dash': (None, None)}  # [1,2,1,10]}
        }

    if ax.get_legend() is not None:
        lines = ax.get_lines() + ax.get_legend().get_lines()
    else:
        lines = ax.get_lines()

    for line in lines:
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)


def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)
