import time
import math
import numpy as np
import os
import json
# Monkey patch the method.
import sys

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Assume the module is one level up from Test_scripts:
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path so Python can find DecoderBlock.py
sys.path.insert(0, project_root)
from DecoderBlock import DecoderBlock as DB 
from Server import Server
from Communicationlink import CommunicationLink as CL
from MIP_opti import OptiProblem

def init_Opti_Problem():
    servers = []
    comm_links = []
    num_servers = 9

    for i in range(num_servers):
        server = Server(
            server_id=i,
            base_throughput=100,
            degrading_factor=0.1,
            selection_strategy="round_robin"
        )
        servers.append(server)

    for i in range(num_servers):
        for j in range(num_servers):
            if i != j:
                comm_link = CL(
                    link_id=len(comm_links),
                    from_entity=servers[i],
                    to_entity=servers[j],
                    data_rate_bps=1e6,
                    speed_of_signal_m_s=3e8,
                    physical_length_m=100
                )
                comm_links.append(comm_link)

    opti_problem = OptiProblem(
        servers=servers,
        comm_links=comm_links,
        max_L=4,
        num_decoder_blocks=4,
        SL_max=10,
        d_model=512,
        precision=16,
        epsilon=None
    )

    path, cost = opti_problem.solve()
    print(f"Optimal path: {path}")
    print(f"Optimal cost: {cost}")

    X, Y, Z = opti_problem.translate(path[0], path[1], path[2]) 
    print(f"X: {X}")
    print(f"Y: {Y}")
    print(f"Z: {Z}")

    



def main():
    init_Opti_Problem()

if __name__ == "__main__":
    main()