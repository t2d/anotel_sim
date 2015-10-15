import csv
import sim
import multiprocessing
import operator
from multiprocessing import Pool


filename = 'sim3d.csv'


def worker3d(users, events, rounds):
    ''' make rounds of simulation of users '''
    print "{0} users with {1} events started".format(users, events)
    paths = []
    for _ in range(rounds):
        log, attacker_log = sim.simulate(users, messages=events)
        path = sim.match_paths(log, attacker_log)
        paths.append(path)
    # build mean and confidence interval
    mean, conf = sim.mean_confidence_interval(paths)
    print "{0} users with {1} events finished".format(users, events)
    return users, events, mean


def log_result(result):
    # export data to csv
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(list(result))
        f.close()


def sort_csv(file):
    f = open(file)
    data = csv.reader(f)
    sortedlist = sorted(data, key=lambda x: (int(x[0]), int(x[1])))
    f.close()

    # now write the sorted result into same file
    with open(file, "w") as f:
        file_writer = csv.writer(f)
        for row in sortedlist:
            file_writer.writerow(row)
    f.close()


def main(max_users=2500, max_events=120, step_user=50, step_events=10, rounds=50):
    # change users and events
    # only complete paths
    users = range(0, max_users+1, step_user)
    events = range(0, max_events+1, step_events)
    users[0] = 1
    events[0] = 1

    # make all cases
    cases = [(u, e) for u in reversed(users) for e in reversed(events)]

    # see if already computed
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                u = int(row[0])
                e = int(row[1])
                if (u, e) in cases:
                    cases.remove((u, e))
                    print("skip ({0}, {1})".format(u, e))
            f.close()
    except (OSError, IOError):
        pass

    # start jobs for processes
    processes = min(len(cases), multiprocessing.cpu_count()-1)
    pool = Pool(processes)
    results = [pool.apply_async(worker3d, args=(u, e, rounds), callback=log_result) for (u, e) in cases]
    pool.close()
    pool.join()

    sort_csv(filename)


if __name__ == "__main__":
    main()
