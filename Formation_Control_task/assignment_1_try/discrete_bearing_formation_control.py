from Formation_Control_task.Formation_Control_task_ROS2.src.formation_task.formation_task.functions import *

# Plot Settings - to impose usage of LaTeX as text renderer
FIGURE = True

NN = 4  # number of agents
dd = 2  # dimension of the pos/vel vector (2-dimensional or 3-dimension motion)
n_leaders = 2  # number of leaders
p_ER = 0.9
n_follower = NN - n_leaders  # number of followers

dt = 10e-4 # discretization step
T_max = 100  # max simulation time
horizon = np.linspace(0.0, T_max, int(T_max / dt))
max_iter = 500  # max number of iterations

# gain parameters
k_p = 0.7
k_v = 1.5

Node_pos = np.zeros((NN, dd, 1))  # it stores all the vector position for each node
# it is a tensor with NN matrix of dimension (dxd)
# to extract a specific vector position you should do Node_pos[i-th node,:,:]

# try to reproduce a rectangle
# Node 0, ( px py ).T
Node_pos[0, :, :] = np.array([
    [1, 5]
]).T
# Node 1 (px py).T
Node_pos[1, :, :] = np.array([
    [5, 5]
]).T
# Node 2 (px py).T
Node_pos[2, :, :] = np.array([
    [1, 1]
]).T
# Node 3 (px py).T
Node_pos[3, :, :] = np.array([
    [5, 1]
]).T

# Graph and Adjacency matrix
G = nx.binomial_graph(NN, p_ER)
Adj = nx.to_numpy_array(G)

# Stack of all Pg_ij
Pg_stack = proj_stack(Node_pos, NN, dd)
# Bearing Laplacian Matrix
B = bearing_laplacian(Pg_stack, Adj, dd)

if FIGURE:
    plt.figure(1)
    nx.draw(G, with_labels=True)
    plt.show()

### TEST ###

Bff = B[dd * n_leaders:dd * NN, dd * n_leaders:dd * NN]
Bfl = B[dd * n_leaders:dd * NN, :dd * n_leaders]

pos_leaders = Node_pos[[0, 1], :, :].reshape([4, 1])
if np.linalg.det(Bff) != 0:
    print("The matrix Bff is not singular")
    pf_star = - np.linalg.inv(Bff) @ Bfl @ pos_leaders
else:
    print("The matrix Bff is singular! Check the Graph")
###


## Initialization of state vector, reference vector and init vector
xx = np.zeros((NN * dd * 2, len(horizon)))
xx_star = np.zeros((NN * dd * 2, len(horizon)))
xx_init = np.zeros((NN * 2 * dd, 1))

# Kroenecker Function of the dynamic
A_kron = kron_dynamical_matrix(B, NN, n_leaders, k_p, k_v, dd)

# We impose the leaders position
for i in range(NN):
    for j in range(dd):
        xx_star[(n_leaders * i) + j] = Node_pos[i, j, :]

### ???? Non l'hai già fatto prima?
for i in range(n_leaders):
    for j in range(dd):
        xx_init[(n_leaders * i) + j] = Node_pos[i, j, :]

xx[:, 0] = xx_init.reshape(16)

for tt in range(len(horizon) - 1):
    xx[:, tt+1] = xx[:, tt] + dt*(A_kron @ xx[:, tt])

if FIGURE:
    fig2 = plt.figure(2)
    plt.title("Evolution of $p_y$")
    plt.xlabel("time")
    plt.ylabel("$p_y$")
    plt.grid(visible=True)
    for i in range(NN):
        plt.plot(horizon, xx[i * 2, :])
        plt.plot(horizon, xx_star[i * 2, :], "r--", linewidth=0.5)

    fig3 = plt.figure(3)
    plt.title("Evolution of $p_x$")
    plt.xlabel("time")
    plt.ylabel("$p_x$")
    plt.grid(visible=True)
    for i in range(NN):
        plt.plot(horizon, xx[i * 2 + 1, :])
        plt.plot(horizon, xx_star[i * 2 + 1, :], "r--", linewidth=0.5)

    fig4 = plt.figure(4)
    plt.title("Evolution of Nodes position in XY-plane")
    plt.xlabel("$p_x$")
    plt.ylabel("$p_y$")
    plt.grid(visible=True)
    for i in range(NN):
        plt.plot(xx[i * 2, :], xx[i * 2 + 1, :], "r--", linewidth=0.5)
        if i < n_leaders:
            plt.plot(xx[i * 2, len(horizon) - 1], xx[i * 2 + 1, len(horizon) - 1], "ro")  # leaders in red
        else:
            plt.plot(xx[i * 2, len(horizon) - 1], xx[i * 2 + 1, len(horizon) - 1], "bo") # followers in blue

print("DAJE TUTTO OK")
