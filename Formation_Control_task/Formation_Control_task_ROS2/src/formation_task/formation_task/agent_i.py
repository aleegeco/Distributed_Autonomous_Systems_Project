#
# 11/05/2022
# IN-LP
#
from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as msg_float


#def formation_update(dt, x_i, neigh, data, dist):

    # xdot_i = np.zeros(x_i.shape)
    #
    # for j in neigh:
    #     x_j = np.array(data[j].pop(0)[1:])
    #     dV_ij = ((x_i - x_j).T@(x_i - x_j) - dist[j]**2)*(x_i - x_j)
    #     xdot_i += - dV_ij

def update_dynamics(dt, x_i, neigh, data, pos, A_kron, agent_id):
    n_x = x_i.shape
    xdot_i = np.zeros(n_x)
    node_i = agent_id
    index_ii = node_i * n_x + np.arange(n_x)

    for node_j in neigh:
        x_j = np.array(data[node_j].pop(0)[1:])
        index_jj = node_j*n_x + np.arange(n_x)
        xdot_i = A_kron[index_ii, index_jj]

    x_i += dt*xdot_i

    return x_i


def writer(file_name, string):
    """
      inner function for logging
    """
    file = open(file_name, "a") # "a" is for append
    file.write(string)
    file.close()

class Agent(Node):
    def __init__(self):
        super().__init__('agent',
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)
            
        # Get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value # index of the agent
        self.neigh = self.get_parameter('neigh').value # list  of neighbors
        A_kron = self.get_parameter('bearing_laplacian').value # bearing laplacian from launcher
        self.A_kron = np.array(A_kron) # it gives the right shape to the matrix A_kron (d*NN, d*NN)
        
        pos_ii = self.get_parameter('pos_xy').value # position vector of the node_i
        self.pos_ii = np.array(pos_ii) # it returns a dd by 1 array

        x_i = self.get_parameter('x_init').value # state vector value of node_i
        self.n_x = len(x_i) # dimension of the state vector of node_i
        self.x_i = np.array(x_i) # it returns an n_x by 1 array
        
        self.max_iters = self.get_parameter('max_iters').value
        self.communication_time = self.get_parameter('communication_time').value

        self.tt = 0

        # create logging file
        self.file_name = "_csv_file/agent_{}.csv".format(self.agent_id)
        file = open(self.file_name, "w+") # 'w+' needs to create file and open in writing mode if doesn't exist
        file.close()

        # initialize subscription dict
        self.subscriptions_list = {}

        # create a subscription to each neighbor
        for j in self.neigh:
            topic_name = '/topic_{}'.format(j)
            self.subscriptions_list[j] = self.create_subscription(
                                                                msg_float, 
                                                                topic_name, 
                                                                lambda msg, node = j: self.listener_callback(msg, node), 
                                                                10)
        
        # create the publisher
        self.publisher_ = self.create_publisher(
                                                                msg_float, 
                                                                '/topic_{}'.format(self.agent_id),
                                                                10)

        self.timer = self.create_timer(self.communication_time, self.timer_callback)

        # initialize a dictionary with the list of received messages from each neighbor j [a queue]
        self.received_data = { j: [] for j in self.neigh }

        print("Setup of agent {} complete".format(self.agent_id))

    def listener_callback(self, msg, node):
        self.received_data[node].append(list(msg.data))

    def timer_callback(self):
        # Initialize a message of type float
        msg = msg_float()

        if self.tt == 0: # Let the publisher start at the first iteration
            msg.data = [float(self.tt)]

            for element in self.x_i:
                msg.data.append(float(element))
            #[msg.data.append(float(element)) for element in self.x_i]

            self.publisher_.publish(msg)
            self.tt += 1

            # log files
            # 1) visualize on the terminal
            string_for_logger = [round(i,4) for i in msg.data.tolist()[1:]]
            print("Iter = {} \t Value = {}".format(int(msg.data[0]), string_for_logger))

            # 2) save on file
            data_for_csv = msg.data.tolist().copy()
            data_for_csv = [str(round(element,4)) for element in data_for_csv[1:]]
            data_for_csv = ','.join(data_for_csv)
            writer(self.file_name,data_for_csv+'\n')

        else: 
            # Check if lists are nonempty
            all_received = all(self.received_data[j] for j in self.neigh) # check if all neighbors' have been received

            sync = False
            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(self.tt-1 == self.received_data[j][0][0] for j in self.neigh) # True if all True

            if sync:
                DeltaT = self.communication_time/10
                self.x_i = update_dynamics(DeltaT, self.x_i, self.neigh, self.received_data, self.pos_ii, self.A_kron,
                                           self.agent_id)
                
                # publish the updated message
                msg.data = [float(self.tt)]
                [msg.data.append(float(element)) for element in self.x_i]
                self.publisher_.publish(msg)

                # save data on csv file
                data_for_csv = msg.data.tolist().copy()
                data_for_csv = [str(round(element,4)) for element in data_for_csv[1:]]
                data_for_csv = ','.join(data_for_csv)
                writer(self.file_name,data_for_csv+'\n')

                string_for_logger = [round(i,4) for i in msg.data.tolist()[1:]]
                print("Iter = {} \t Value = {}".format(int(msg.data[0]), string_for_logger))
                
                # Stop the node if tt exceeds MAXITERS
                if self.tt > self.max_iters:
                    print("\nMAXITERS reached")
                    sleep(3) # [seconds]
                    self.destroy_node()

                # update iteration counter
                self.tt += 1

def main(args=None):
    rclpy.init(args=args)

    agent = Agent()
    print("Agent {:d} -- Waiting for sync.".format(agent.agent_id))
    sleep(0.5)
    print("GO!")

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        print("----- Node stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()