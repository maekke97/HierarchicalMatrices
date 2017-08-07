import matplotlib.pyplot as plt
import csv

nmin = 12
figsize = (8, 6)
dpi = 200

infile = '/compute/nem/rmat_addition_profiler.csv'
values = {}
with open(infile, 'r') as csvfile:
    value_reader = csv.reader(csvfile, delimiter=',')
    for row in value_reader:
        n = int(row[0])
        k = int(row[1])
        secs = float(row[2])
        values[(n, k)] = secs

fig = plt.figure(figsize=figsize, dpi=dpi)
plt.title('Formatted addition with rank fixed')
xs = [2**i for i in xrange(nmin, nmin + 10)]
for k in xrange(1, 11):
    ax = fig.add_subplot(2, 5, k)
    times = [values[(x, k)] for x in xs]
    ax.plot(xs, times)
    plt.title('k={0}'.format(k))
plt.tight_layout()

plt.savefig('rmat_addition_k_profile.png')

fig = plt.figure(figsize=figsize, dpi=dpi)
plt.title('Formatted addition with n fixed')
ks = range(1, 11)
for i in xrange(nmin, nmin + 10):
    ax = fig.add_subplot(2, 5, i - nmin + 1)
    times = [values[(2**i, k)] for k in ks]
    ax.plot(ks, times)
    plt.title('n={0}'.format(2**i))
plt.tight_layout()

plt.savefig('rmat_addition_n_profile.png')


infile = '/compute/nem/rmat_subtraction_profiler.csv'
values = {}
with open(infile, 'r') as csvfile:
    value_reader = csv.reader(csvfile, delimiter=',')
    for row in value_reader:
        n = int(row[0])
        k = int(row[1])
        secs = float(row[2])
        values[(n, k)] = secs

fig = plt.figure(figsize=figsize, dpi=dpi)
plt.title('Formatted subtraction with rank fixed')
xs = [2**i for i in xrange(nmin, nmin + 10)]
for k in xrange(1, 11):
    ax = fig.add_subplot(2, 5, k)
    times = [values[(x, k)] for x in xs]
    ax.plot(xs, times)
    plt.title('k={0}'.format(k))
plt.tight_layout()

plt.savefig('rmat_subtraction_k_profile.png')

fig = plt.figure(figsize=figsize, dpi=dpi)
plt.title('Formatted subtraction with n fixed')
ks = range(1, 11)
for i in xrange(nmin, nmin + 10):
    ax = fig.add_subplot(2, 5, i - nmin + 1)
    times = [values[(2**i, k)] for k in ks]
    ax.plot(ks, times)
    plt.title('n={0}'.format(2**i))
plt.tight_layout()

plt.savefig('rmat_subtraction_n_profile.png')

infile = '/compute/nem/rmat_multiplication_profiler.csv'
values = {}
with open(infile, 'r') as csvfile:
    value_reader = csv.reader(csvfile, delimiter=',')
    for row in value_reader:
        n = int(row[0])
        k = int(row[1])
        secs = float(row[2])
        values[(n, k)] = secs

fig = plt.figure(figsize=figsize, dpi=dpi)
plt.title('Multiplication with rank fixed')
xs = [2**i for i in xrange(nmin, nmin + 10)]
for k in xrange(1, 11):
    ax = fig.add_subplot(2, 5, k)
    times = [values[(x, k)] for x in xs]
    ax.plot(xs, times)
    plt.title('k={0}'.format(k))
plt.tight_layout()

plt.savefig('rmat_multiplication_k_profile.png')

fig = plt.figure(figsize=figsize, dpi=dpi)
plt.title('Formatted multiplication with n fixed')
ks = range(1, 11)
for i in xrange(nmin, nmin + 10):
    ax = fig.add_subplot(2, 5, i - nmin + 1)
    times = [values[(2**i, k)] for k in ks]
    ax.plot(ks, times)
    plt.title('n={0}'.format(2**i))
plt.tight_layout()

plt.savefig('rmat_multiplication_n_profile.png')
