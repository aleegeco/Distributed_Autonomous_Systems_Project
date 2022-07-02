from roboticstoolbox.tools.trajectory import *
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0,10,)
tg = quintic(0,5,t)
acceleration = tg.qdd
tg.plot()



