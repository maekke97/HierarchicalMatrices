import matplotlib.pyplot as plt
import csv


infile = '/compute/nem/rmat_addition_profiler.csv'
values = {}
with open(infile, 'r') as csvfile:
    value_reader = csv.reader(csvfile, delimiter=',')
    for row in value_reader:
        n = int(row[0])
        k = int(row[1])
        secs = float(row[2])
        values[(n, k)] = secs

fig = plt.figure()
plt.title('Formatted addition with rank fixed')
xs = [2**i for i in xrange(4, 25)]
for k in xrange(1, 11):
    ax = fig.add_subplot(2, 5, k)
    times = [values[(x, k)] for x in xs]
    ax.plot(xs, times)
    plt.title('k={0}'.format(k))

plt.show()

fig = plt.figure()
plt.title('Formatted addition with n fixed')
ks = range(1, 11)
for i in xrange(4, 25):
    ax = fig.add_subplot(3, 7, i-3)
    times = [values[(2**i, k)] for k in ks]
    ax.plot(ks, times)
    plt.title('n={0}'.format(2**i))

plt.show()