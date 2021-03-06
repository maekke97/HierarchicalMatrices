#!/usr/bin/env python
import matplotlib.pyplot as plt
import math

ct_secs_old = [0.001, 0.005, 0.007, 0.013, 0.031, 0.066, 0.144, 0.314, 0.618, 1.553, 3.115, 6.208, 13.39, 25.74, 53.16, 118.4, 237.4, 510.9, 1103, 2327, 4769, 11261, 28495]
ct_mem_old = [0.039, 0.051, 0.062, 0.062, 0.094, 0.281, 0.820, 1.871, 4.031, 8.395, 17.15, 34.40, 69.07, 138.8, 278.7, 559.6, 1123, 2254, 4524, 9080, 18224, 36582, 73424]
bct_secs_old = [0.006, 0.027, 0.070, 0.173, 0.362, 0.756, 1.606, 3.184, 6.151, 12.42, 22.79, 46.12, 95.92, 170.8, 348.9, 716.5, 1435, 2818, 5866, 12412, 28439, 62429, 169554]
bct_mem_old = [0.016, 0.020, 0.008, 0.012, 0.195, 0.527, 1.086, 2.172, 4.352, 8.699, 17.41, 34.83, 70.67, 139.3, 278.6, 557.1, 1114, 2230, 4457, 8914, 17828, 35658, 71330]

ct_secs_new = [0.003, 0.004, 0.008, 0.018, 0.039, 0.084, 0.180, 0.393, 0.841, 1.743, 3.789, 7.926, 15.30, 31.51, 64.66, 135.1, 281.9, 592.9, 1307, 2733, 5697, 11658, 25146]
ct_mem_new = [0.0, 0.0, 0.0, 0.3, 0.4, 0.5, 1.1, 1.9, 4.0, 7.9, 15.6, 30.8, 61.9, 124.4, 249.1, 499.3, 1000, 2004, 4021, 8055, 16146, 32364, 64859]
bct_secs_new = [0.012, 0.033, 0.080, 0.175, 0.407, 0.772, 1.587, 3.167, 6.497, 12.05, 22.99, 44.24, 98.10, 178.0, 357.1, 792.5, 1524, 3291, 6442, 13297, 26374, 55677, 123721]
bct_mem_new = [0.0, 0.3, 0.4, 0.3, 0.8, 1.9, 3.8, 7.7, 15.4, 30.8, 61.5, 123.4, 148.0, 494.3, 988.8, 1979, 3957, 7914, 15824, 31651, 63299, 126592, 252873]

xs = [2**i for i in xrange(2, 25)]
ns = [float(x)/500 for x in xs]
ns_low = [float(x)/1000 for x in xs]

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)

ax.plot(xs, ct_secs_old)
ax.plot(xs, ct_secs_new)
ax.plot(xs, ns, 'k-')
ax.plot(xs, ns_low, 'g-')
plt.title('ClusterTree seconds')
plt.legend(['old', 'new', 'n/500', 'n/1000'])

ax = fig.add_subplot(2, 2, 2)

ns = [float(x)/200 for x in xs]
ns_low = [float(x)/300 for x in xs]

ax.plot(xs, ct_mem_old)
ax.plot(xs, ct_mem_new)
ax.plot(xs, ns, 'k-')
ax.plot(xs, ns_low, 'g-')
plt.title('ClusterTree memory')
plt.legend(['old', 'new', 'n/100', 'n/200'])

ax = fig.add_subplot(2, 2, 3)

ns = [float(x)/90 for x in xs]
ns_low = [float(x)/200 for x in xs]

ax.plot(xs, bct_secs_old)
ax.plot(xs, bct_secs_new)
ax.plot(xs, ns, 'k-')
ax.plot(xs, ns_low, 'g-')
plt.title('BlockClusterTree seconds')
plt.legend(['old', 'new', 'n/90', 'n/200'])

ax = fig.add_subplot(2, 2, 4)

ns = [float(x)/50 for x in xs]
ns_low = [float(x)/300 for x in xs]

ax.plot(xs, bct_mem_old)
ax.plot(xs, bct_mem_new)
ax.plot(xs, ns, 'k-')
ax.plot(xs, ns_low, 'g-')
plt.title('BlockClusterTree memory')
plt.legend(['old', 'new', 'n/50', 'n/300'])

plt.show()
