from roboticstoolbox.tools.trajectory import *
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0,1000,1000)
tg = quintic(0,100,t)
acceleration = 500*tg.qdd

plt.figure()
plt.plot(t,acceleration,LineWidth=1.5)
plt.title("Acceleration Profile")
plt.ylabel("$\dot{v}(t)$")
plt.xlabel("time")
ax = plt.gca()
ax.set_facecolor('w')
plt.show()
