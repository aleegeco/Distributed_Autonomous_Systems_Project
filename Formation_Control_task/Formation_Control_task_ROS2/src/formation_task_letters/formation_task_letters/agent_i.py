
from time import sleep
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as msg_float
from formation_task_letters.functions_letters import *


def writer(file_name, string):
    file = open(file_name, "a") # "a" stands for "append"
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

        self.k_p = self.get_parameter('k_p').value
        self.k_v = self.get_parameter('k_v').value
        self.n_leaders = self.get_parameter('n_leaders').value
        self.NN = self.get_parameter('n_agents').value

        x_i = self.get_parameter('xx_init').value # state vector value of node_i
        self.n_x = len(x_i) # dimension of the state vector of node_i
        self.x_i = np.array(x_i) # it returns an n_x by 1 array
        dd = self.n_x//2

        formation = self.get_parameter('formation').value
        self.formation = np.array(formation).reshape(3,self.NN,dd)

        self.max_iters = self.get_parameter('max_iters').value
        self.communication_time = self.get_parameter('communication_time').value

        self.tt = 0
        self.start = 0

        # create logging file
        self.file_name = "_csv_file/agent_{}.csv".format(self.agent_id)
        # self.file_name_pos = "_csv_file_pos/agent_ref_pos{}.csv".format(self.agent_id)
        file = open(self.file_name, "w+") # 'w+' needs to create file and open in writing mode if doesn't exist
        # file_pos = open(self.file_name_pos, "w+")
        file.close()
        # file_pos.close()

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

            [msg.data.append(float(element)) for element in self.x_i]
            msg.data.append(self.start)
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
            writer(self.file_name, data_for_csv + '\n')

            # # 3) csv file for nodes reference positions
            # data_pos_csv = self.node_ref_pos[self.agent_id,:,:].tolist().copy() #TODO aggiornare csv per nuova formation
            # data_pos_csv = [str(element) for element in data_pos_csv]
            # data_pos_csv = ','.join(data_pos_csv)
            # writer(self.file_name_pos, data_pos_csv + '\n')

        else: 
            # Check if lists are nonempty
            all_received = all(self.received_data[j] for j in self.neigh) # check if all neighbors' have been received

            sync = False
            # Have all messages at time t-1 arrived?
            if all_received:
                sync = all(self.tt-1 == self.received_data[j][0][0] for j in self.neigh) # True if all True

            if sync:
                DeltaT = self.communication_time/10
                self.x_i, self.start = update_dynamics(DeltaT, self.x_i, self.neigh, self.received_data, self.formation,
                                           self.agent_id, self.n_leaders, self.k_p, self.k_v)
                
                # publish the updated message
                msg.data = [float(self.tt)]
                [msg.data.append(float(element)) for element in self.x_i]
                msg.data.append(self.start)
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