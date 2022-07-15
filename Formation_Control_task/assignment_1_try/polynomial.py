from roboticstoolbox.tools.trajectory import *
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
iter = 500
t = np.linspace(0,iter,iter)
delta = 2-4
tg0 = quintic(0,2,t).PLOT()
tg1 = quintic(2,2,t)
tg2 = quintic(2,6,t)
tg3 = quintic(6,8,t)

# tg = np.concatenate((tg0.qdd, tg1.qdd, tg2.qdd, tg3.qdd))
# plt.figure()
# plt.PLOT(tg)
# plt.show()

# plt.figure()
# plt.PLOT(t,acceleration,LineWidth=1.5)
# plt.title("Acceleration Profile")
# plt.ylabel("$\dot{v}(t)$")
# plt.xlabel("time")
# ax = plt.gca()
# ax.set_facecolor('w')
# plt.show()

type(ctrl.saturation_nonlinearity(1, 0))
