from numpy import linspace, cos, pi, ceil, floor, arange, sin, sinc
from matplotlib.pyplot import plot, show, axis, subplot, errorbar
import random

f = 1.0
fs = 10.0

t = linspace(-1, 1, 100)
ts = arange(-1, 1+1/fs, 1/fs)
ts = random.random()

wsp = len(ts)
s_rec = 0

# a = random.random()

for k in range(-wsp, wsp):
    s_rec += sin(2 * pi * (k/fs)) * sinc(k - fs*t)

# plot
plot(t, s_rec, '--', t, sin(2 * pi * t), ts, sin(2 * pi * ts), 'o')

show()

blad = (s_rec - sin(2 * pi * t))**2

plot(blad)
show()
