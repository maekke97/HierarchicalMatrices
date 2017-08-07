import matplotlib.pyplot as plt
import csv


infile = '/compute/nem/rmat_profiler.csv'
values = {}
with open(infile, 'r') as csvfile:
    value_reader = csv.reader(csvfile, delimiter=',')
    for row in value_reader:
        n = int(row[0])
        k = int(row[1])
        secs = float(row[2])
        values[(n, k)] = secs

fig = plt.figure()
xs = [2**i for i in xrange(4, 25)]
for k in xrange(1, 11):
    ax = fig.add_subplot(2, 5, k)
    times = [values[(x, k)] for x in xs]
    ax.plot(xs, times)
    plt.title('k={0}'.format(k))

plt.show()
