import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from multiprocessing import Pool
import scipy as sp
import scipy.stats
import sys
import csv
import operator
from math import sqrt, floor
from random import randint, uniform, seed
from pylab import *


n_cells = 2500
side = 50
n_areas = 42
seconds = 120 * 60
# 54 events per day per person

# create logs
ANOTEL = 1
TRADITIONAL = 2
CELL = 3
MOVEMENT = 4
DEBUG = 5

LOGLEVEL = 0

xcuts = None
ycuts = None


class Operator():
    def __init__(self):
        self.cells_to_area = dict()
        self.areas = dict()

        # cut grid in areas randomly
        self.xcuts = Operator.cut_grid(6)  # 7*6=42
        self.ycuts = Operator.cut_grid(7)
        global xcuts, ycuts
        xcuts = [x+1 for x in self.xcuts]
        ycuts = [y+1 for y in self.ycuts]

        for y in self.ycuts:
            for x in self.xcuts:
                name = self.xcuts.index(x) + 6*self.ycuts.index(y)
                self.areas[name] = (x, y)

        # assign cells to areas
        for c in range(n_cells):
            x = c % 50
            y = c / 50

            for name, coord in self.areas.items():
                if x <= coord[0]:
                    if y <= coord[1]:
                        self.cells_to_area[c] = name
                        break

    @staticmethod
    def cut_grid(amount):
        ''' cuts the grid into location areas '''
        cuts = [49]
        while len(cuts) < amount:
            cut = randint(2, 47)
            if cut not in cuts and cut+1 not in cuts and cut-1 not in cuts:
                cuts.append(cut)
        cuts.sort()
        return cuts

    @staticmethod
    def get_cell(pos):
        x, y = pos
        index = 50*floor(y)+floor(x)
        return index

    @staticmethod
    def get_cell_coordinates(cell_id):
        x = floor(cell_id % 50)
        y = floor(cell_id / 50)
        return (x, y)


class User():
    def __init__(self, name, start, stop, operator, now, logs, level, n_events):
        self.name = name
        self.op = operator
        self.start = start
        self.pos = self.start
        self.goal = stop
        self.cell = self.op.get_cell(self.start)
        self.area = self.get_area()
        self.x_speed = (self.goal[0] - self.start[0]) / seconds
        self.y_speed = (self.goal[1] - self.start[1]) / seconds
        self.time = now
        self.logs = logs
        self.log_level = level
        reason = "%s start" % (self.pos,)
        self.log(reason, ANOTEL)

        # create events at random time and distribute equally
        self.outgoing_events = []
        self.incoming_events = []
        events = []
        while len(events) < n_events:
            event = randint(1, seconds)
            if event not in events:
                events.append(event)
        for e in events:
            # fuer meta message sim
            # if events.index(e) % 2 == 0:
            self.outgoing_events.append(e)
            # else:
            #    self.incoming_events.append(e)

    def move(self, now):
        self.time = now
        if self.pos != self.goal:
            x = self.pos[0] + self.x_speed
            y = self.pos[1] + self.y_speed
            self.pos = (x, y)
            self.log(str(self.pos), MOVEMENT)

            # output cell change
            cell = self.op.get_cell(self.pos)
            if cell != self.cell:
                reason = "cell %d to cell %d" % (self.cell, cell)
                self.cell = cell
                self.log(reason, CELL)
                area = self.get_area()
                if area != self.area:
                    reason = "area %d to area %d" % (self.area, area)
                    self.log(reason, ANOTEL)
                    self.area = area

            # log goal or calc next steps
            if self.pos == self.goal:
                reason = "%s goal" % (self.pos,)
                self.log(reason, MOVEMENT)

    def live(self, now):
        self.move(now)
        if now in self.incoming_events:
            self.log("Incoming event", TRADITIONAL)
        if now in self.outgoing_events:
            self.log("Outgoing event", ANOTEL)

    def get_area(self):
        return self.op.cells_to_area[self.cell]

    def log(self, reason, level):
        coordinates = self.op.get_cell_coordinates(self.cell)
        if level <= self.log_level:
            print "%s %s\tat %d\t# %s" % (self.name, coordinates, self.time, reason)

        # cell log
        (log_x, log_y) = self.logs[MOVEMENT][self.name]
        log_x.append(self.pos[0])
        log_y.append(self.pos[1])

        if level <= TRADITIONAL:
            (log_x, log_y) = self.logs[TRADITIONAL][self.name]
            log_x.append(coordinates[0])
            log_y.append(coordinates[1])

        if level <= ANOTEL:
            (log_x, log_y) = self.logs[ANOTEL]
            log_x.append(coordinates[0])
            log_y.append(coordinates[1])


class Attacker():
    def __init__(self, log, output):
        self.log = log
        self.output = output
        self.used_log = None
        self.users = output
        self.max_speed = sqrt(2*n_cells) / seconds

    def attack(self, now):
        x_log = self.log[0]
        y_log = self.log[1]
        assert(len(x_log) == len(y_log))

        # start logging
        if not self.users:  # if empty
            for i in range(len(x_log)):
                self.users[i] = ([x_log[i]], [y_log[i]], [now])  # x, y, t
        else:
            # get new coordinate pairs
            used = len(self.used_log[0])
            new_x = x_log[used:]
            new_y = y_log[used:]
            assert(len(new_x) == len(new_y))

            # append to user with minimal distance
            for x in new_x:
                y = new_y[new_x.index(x)]
                min_dist = 999999999  # just very high value
                min_user = None

                for user, log in self.users.iteritems():
                    old_x = log[0][-1]
                    old_y = log[1][-1]
                    dist = abs(sqrt((x - old_x)**2 + (y - old_y)**2))

                    if dist < min_dist:
                        min_dist = dist
                        min_user = user

                if min_user is not None:
                    self.users[min_user][0].append(x)
                    self.users[min_user][1].append(y)
                    self.users[min_user][2].append(now)
                else:
                    print "ERROR: Didn't find user for (%d,%d)" % (x, y)

        # refresh used_log
        self.used_log = (list(x_log), list(y_log))


def simulate(n_users, level=LOGLEVEL, messages=6):
    if n_users > n_cells:
        print "Not possible atm. Too many users"
        sys.exit(1)
    # create logs
    logs = {TRADITIONAL: {}, MOVEMENT: {}}
    for key in logs:
        for i in range(n_users):
            logs[key][i] = ([], [])
    # no user-specific log in ANOTEL
    logs[ANOTEL] = ([], [])

    # create op and attacker
    op = Operator()
    attacker_log = {}
    attacker = Attacker(logs[ANOTEL], attacker_log)

    # create users
    users = []
    distinct_cells = set()
    length = sqrt(n_cells)
    name = 0

    while len(users) < n_users:
        start = (uniform(0, length), uniform(0, length))
        stop = (uniform(0, length), uniform(0, length))
        start_cell = Operator.get_cell(start)
        stop_cell = Operator.get_cell(stop)
        if start_cell != stop_cell and start_cell not in distinct_cells:
            users.append(User(name, start, stop, op, 0, logs, level, messages))
            distinct_cells.add(start_cell)
            name += 1

    # progress time and move users on grid
    for t in range(seconds):
        for u in users:
            u.live(t+1)
        attacker.attack(t+1)

    return logs, attacker_log


def get_point_combinations(log, length):
    # build combinations of length two or three
    all_points = []

    for user in log:
        points = get_unique_points(user)

        for i in range(len(points)-(length-1)):
            if length == 2:
                all_points.append((points[i], points[i+1]))
            elif length == 3:
                all_points.append((points[i], points[i+1], points[i+2]))

    return all_points


def get_unique_points(log):
    points = []

    for i, x in enumerate(log[0]):
        p = (x, log[1][i])
        if p not in points:
            points.append(p)

    return points


def match(logs, attacker_log):
    all = 0
    points_in_combis = []
    points_in_3combis = []
    starts = 0
    endpoints = 0

    # enumerate all points
    for ano in attacker_log.values():
        ano_points = get_unique_points(ano)
        all += len(ano_points)

    # find sequences
    ano_combis = get_point_combinations(attacker_log.values(), 2)
    for c in ano_combis:
        for user in logs[TRADITIONAL].values():
            user_points = get_unique_points(user)
            if c[0] in user_points and c[1] in user_points:
                if user_points.index(c[0]) < user_points.index(c[1]):
                    # add points to counting array
                    for i in range(2):
                        if c[i] not in points_in_combis:
                            points_in_combis.append(c[i])

    ano_combis = get_point_combinations(attacker_log.values(), 3)
    for c in ano_combis:
        for user in logs[TRADITIONAL].values():
            user_points = get_unique_points(user)
            if c[0] in user_points and c[1] in user_points and c[2] in user_points:
                if user_points.index(c[0]) < user_points.index(c[1]) < user_points.index(c[2]):
                    # add points to counting array
                    for i in range(3):
                        if c[i] not in points_in_3combis:
                            points_in_3combis.append(c[i])

    two = 100 * float(len(points_in_combis))/all
    three = 100 * float(len(points_in_3combis))/all
    paths = match_paths(logs, attacker_log)
    return two, three, paths


def match_paths(logs, attacker_log):
    starts = 0
    endpoints = 0

    # match endpoints
    for ano in attacker_log.values():
        ano_points = get_unique_points(ano)
        # find matching start
        for user in logs[TRADITIONAL].values():
            user_points = get_unique_points(user)
            if user_points[0] == ano_points[0]:
                # users correspondent
                starts += 1
                if user_points[-1] == ano_points[-1]:
                    # endpoint found
                    endpoints += 1

    paths = 100 * float(endpoints)/starts

    return paths


def plot(n_users, logs, attacker_log):
    fig = plt.figure(1)
    make_axes(fig)
    plt.grid()
    for i in range(n_users):
        # movement logs
        # (log_x, log_y) = logs[MOVEMENT][i]
        # plt.plot(log_x, log_y)
        # cell event log
        (log_x, log_y) = logs[TRADITIONAL][i]
        # add 0.5 to add to center of cell
        log_x = [x+0.5 for x in log_x]
        log_y = [y+0.5 for y in log_y]
        plt.scatter(log_x, log_y, c='black')
        plt.plot(log_x, log_y, c='black')
    plt.savefig('normal.svg', format='svg', bbox_inches='tight')

    # anotel plot
    fig2 = plt.figure(2)
    make_axes(fig2)
    plt.grid()
    log_x = [x+0.5 for x in logs[ANOTEL][0]]
    log_y = [y+0.5 for y in logs[ANOTEL][1]]
    plt.plot(log_x, log_y, 'ro', c='black')

    # attacker plot
    # for user in attacker_log.values():
    #     (log_x, log_y, t) = user
    #     # add 0.5 to add to center of cell
    #     log_x = [x+0.5 for x in log_x]
    #     log_y = [y+0.5 for y in log_y]
    #     plt.scatter(log_x, log_y)
    #     plt.plot(log_x, log_y)
    plt.savefig('anotel.svg', format='svg', bbox_inches='tight')


def make_axes(fig):
    axes = fig.gca()
    axes.set_xticks(np.arange(side+1))
    x_ticks = axes.get_xticklabels()
    for label in x_ticks:
        if x_ticks.index(label) % 10 != 0:
            label.set_visible(False)
    axes.set_yticks(np.arange(side+1))
    y_ticks = axes.get_yticklabels()
    for label in y_ticks:
        if y_ticks.index(label) % 10 != 0:
            label.set_visible(False)
    axes.set_autoscaley_on(False)
    axes.set_autoscalex_on(False)
    axes.vlines(xcuts, 0, 50)
    axes.hlines(ycuts, 0, 50)


def worker(step, rounds, mode):
    ''' make rounds of simulation of users '''
    print "{0} started".format(step)
    two = []
    three = []
    paths = []
    for _ in range(rounds):
        if mode == "users":
            log, attacker_log = simulate(step)
        elif mode == "events":
            log, attacker_log = simulate(20, messages=step)
        _2, _3, path = match(log, attacker_log)
        two.append(_2)
        three.append(_3)
        paths.append(path)
    # build mean and confidence interval
    mean_2, conf_2 = mean_confidence_interval(two)
    mean_3, conf_3 = mean_confidence_interval(three)
    mean_p, conf_p = mean_confidence_interval(paths)
    print "{0} finished".format(step)
    return step, mean_2, conf_2, mean_3, conf_3, mean_p, conf_p


def meta_sim(steps, rounds=10, mode="users"):
    # simulate x times for i users
    # try match
    # build mean of endpoints and correct points
    # mode 'events' changes amount of events, not users
    steps.sort()

    # start jobs for processes
    processes = min(len(steps), multiprocessing.cpu_count()-1)
    pool = Pool(processes)
    results = [pool.apply_async(worker, args=(step, rounds, mode)) for step in reversed(steps)]
    pool.close()
    pool.join()

    # collect results
    data = []
    for r in results:
        data.append(r.get())

    sorted_data = sorted(data, key=operator.itemgetter(0))

    # export data to csv
    export = open("metasim_" + mode + ".csv", 'wb')
    writer = csv.writer(export)
    writer.writerows(sorted_data)
    export.close()


def mean_confidence_interval(data, confidence=0.95):
    ''' return mean and confidence intervals of array '''
    # https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t.ppf((1+confidence)/2., n-1)
    return m, h


def main(n_users=10, do_plot=True, seeder=None):
    if seeder is not None:
        seed(seeder)
    log, attacker_log = simulate(n_users)
    two, three, percent_paths = match(log, attacker_log)
    print("%.2f %% 2-point-paths matched" % two)
    print("%.2f %% 3-point-paths matched" % three)
    print("%.2f %% endpoints matched" % percent_paths)
    if do_plot:
        plot(n_users, log, attacker_log)


if __name__ == '__main__':
    main()
