# test_system.py

import os
import random
import networkx as nx
import sys
import copy
# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Assume the module is one level up from Test_scripts:
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path so Python can find DecoderBlock.py
sys.path.insert(0, project_root)

# Assuming your System class is in a file called System.py in the same directory
from System import System
from Server import Server
from Communicationlink import CommunicationLink
from DecoderBlock import DecoderBlock
from Client import Client

def main():
    # Optional: set random seed for reproducibility
    random.seed(42)

    # 1. Create a few servers manually
    #    (Alternatively, you can let System create them if you pass num_servers=... in the constructor)
    servers = []
    for i in range(5):  # for example, 5 servers
        srv = Server(server_id=i)
        # You might configure additional attributes here, e.g., base_throughput, degrading_factor, etc.
        srv.base_throughput = 3.0 + i * 2.0
        srv.degrading_factor = 0.05
        srv.num_blocks = 0  # Make sure to initialize num_blocks for your degrade logic
        servers.append(srv)

    # 2. Create a complete graph of communication links among these servers
    #    That means each server is connected to every other server (both directions).
    comm_links = []
    link_id = 0
    for i in range(len(servers)):
        for j in range(len(servers)):
            if i != j:
                # Create a link from server i to server j
                link = CommunicationLink(
                    link_id=link_id,
                    from_entity=servers[i],
                    to_entity=servers[j],
                )
                comm_links.append(link)
                link_id += 1

    # Initialize a client with id = 1
    clients = []
    client1 = Client(client_id=1, num_stages=2)
    clients.append(client1)

    client2 = Client(client_id=2, num_stages=2)
    clients.append(client2)

    # Create communication links from all servers to all clients
    for server in servers:
        for client in clients:
            link = CommunicationLink(
                link_id=link_id,
                from_entity=server,
                to_entity=client,
            )
            comm_links.append(link)
            link_id += 1

    # Create communication links from all clients to all servers
    for client in clients:
        for server in servers:
            link = CommunicationLink(
                link_id=link_id,
                from_entity=client,
                to_entity=server,
            )
            comm_links.append(link)
            link_id += 1
    # You might configure additional attributes here if needed

    # Optionally, add the client to the system if needed
    # system.add_client(client)
    # 3. Instantiate a System with the created servers and links
    #    We’re passing them in explicitly via servers_list and commlinks_list.
    #    No clients are passed here, but you can create some if needed.


    # 4. Create some decoder blocks to assign. Suppose we want 6 blocks total.
    decoder_blocks = []
    for block_id in range(6):
        block = DecoderBlock(block_id=block_id, config_filename='config_simulation.json')
        decoder_blocks.append(block)

    system = System(
        system_id=0,
        system_init="server_commlink_object_based",
        servers_list=servers,
        commlinks_list=comm_links,
        clients_list=clients,
        DecoderBlock_list=decoder_blocks,
        config_filename='config_simulation.json'
        )    

    # 5. Call create_throughput_balanced_swarms
    #    For example, we want 2 swarms total.
    system.make_system_init()

    system.run_time_steps(steps=10000)
    print("----- Finished Throughput-Balanced Swarm Creation -----")

if __name__ == "__main__":
    main()