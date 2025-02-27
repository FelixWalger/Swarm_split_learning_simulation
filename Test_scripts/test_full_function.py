# test_full_function.py

import os
import json
import networkx as nx
import sys
import copy
# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Assume the module is one level up from Test_scripts:
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path so Python can find DecoderBlock.py
sys.path.insert(0, project_root)
from Server import Server
from Communicationlink import CommunicationLink
from Client import Client
from DecoderBlock import DecoderBlock
from Swarm import Swarm

def create_test_config():
    """
    Creates a simple JSON config file for servers, network, and client properties.
    """
    config = {
    "num_systems": 3,
    "num_servers": 5,
    "num_clients": 3,
    "total_time_quantiles": 10,

    "server_properties": {
    "throughput_range": [50, 150],
    "memory_range_in_GB": [0.5, 5],
    "dropout_prob_range": [0.01, 0.1],
    "selection_strategy": "longest_time_not_seen",     
    "down_server_time_range_min": [1, 4],
    "up_server_time_range_min": [10, 25],
    "degradation_rate": [0, 0.1]
    },
    "client_properties": {
    "token_length_range": [3, 8],
    "prompt_mode": "all_together"
    },
    "system_properties": {
    "max_sequence_length": 1024,
    "memory_mapping_token_range": [256, 512]
    },
    "network_properties": {
    "data_rate_range_Mbits/s": [25, 1500],
    "distance_range_km": [0, 30000],  
    "propagation_speed_km/s": 200000,
    "jitter_lognorm_param_tuples_mu_sigma": [[0.1, 0.1], [0.1, 5],[5,0.1]],
    "down_link_time_range_min": [1, 4],
    "up_link_time_range_min": [20, 75]
    },

    "routing_policy": "round_robin",
    "Toplogy_policy": "throughput_balanced",
    "addtional_decoder_block_param":0,
    "precision": 16,
    "precision_decoder_block_params": 0,
    "d_model": 512

    }
    with open('test_config.json', 'w') as f:
        json.dump(config, f)

def inizilatazion_of_objects():
    """
    Demonstrates setting up a two-stage topology (two servers per stage),
    creating a client, checking routing with both round_robin and min_cost_max_flow,
    and launching jobs both from the client and externally.
    """
    create_test_config()

    # --- Create Servers ---
    # Stage 1 servers
    server1 = Server(server_id=1, config_filename='test_config.json')
    server2 = Server(server_id=2, config_filename='test_config.json')
    # Stage 2 servers
    server3 = Server(server_id=3, config_filename='test_config.json')
    server4 = Server(server_id=4, config_filename='test_config.json')

    # --- Create Decoder Blocks ---
    # Stage 1 decoder blocks
    # Stage 1 decoder blocks
    decoder_block1_s1 = DecoderBlock(block_id=1, config_filename='test_config.json')
    decoder_block1_s2 = copy.copy(decoder_block1_s1)
    # Stage 2 decoder blocks
    decoder_block2_s4 = DecoderBlock(block_id=2, config_filename='test_config.json')
    decoder_block2_s3 = copy.copy(decoder_block2_s4)
    
    #Init the client
    client = Client(
        client_id=1,
        num_stages=2,
        config_filename='test_config.json'
    )

    # Assign decoder blocks to servers using the assign_decoder_block method
    server1.assign_decoder_block(decoder_block1_s1, allocation=True)
    server2.assign_decoder_block(decoder_block1_s2, allocation=True)
    server3.assign_decoder_block(decoder_block2_s3, allocation=True)
    server4.assign_decoder_block(decoder_block2_s4, allocation=True)

    # --- Create Communication Links ---
    #
    # Client -> Stage 1
    link_c1_s1 = CommunicationLink(link_id=1, from_entity=client, to_entity=server1, config_filename='test_config.json')
    link_c1_s2 = CommunicationLink(link_id=2, from_entity=client, to_entity=server2, config_filename='test_config.json')

    # Stage 1 -> Stage 2
    link_s1_s3 = CommunicationLink(link_id=3, from_entity=server1, to_entity=server3, config_filename='test_config.json')
    link_s1_s4 = CommunicationLink(link_id=4, from_entity=server1, to_entity=server4, config_filename='test_config.json')
    link_s2_s3 = CommunicationLink(link_id=5, from_entity=server2, to_entity=server3, config_filename='test_config.json')
    link_s2_s4 = CommunicationLink(link_id=6, from_entity=server2, to_entity=server4, config_filename='test_config.json')

    # Stage 2 -> SINK (or final exit)
    link_s3_sink = CommunicationLink(link_id=7, from_entity=server3, to_entity=client, config_filename='test_config.json')
    link_s4_sink = CommunicationLink(link_id=8, from_entity=server4, to_entity=client, config_filename='test_config.json')



    # --- Create NetworkX Topology ---
    topology = nx.DiGraph()
    # Client -> Stage 1
    topology.add_edge("client_1", "server_1_in", object=link_c1_s1)
    topology.add_edge("client_1", "server_2_in", object=link_c1_s2)
    # Stage 1 -> Stage 2
    topology.add_edge("server_1_out", "server_3_in", object=link_s1_s3)
    topology.add_edge("server_1_out", "server_4_in", object=link_s1_s4)
    topology.add_edge("server_2_out", "server_3_in", object=link_s2_s3)
    topology.add_edge("server_2_out", "server_4_in", object=link_s2_s4)
    # Stage 2 -> SINK
    topology.add_edge("server_3_out", "SINK_in", object=link_s3_sink)
    topology.add_edge("server_4_out", "SINK_in", object=link_s4_sink)
    # Internal edges for servers
    topology.add_edge("server_1_in", "server_1_out", object=server1)
    topology.add_edge("server_2_in", "server_2_out", object=server2)
    topology.add_edge("server_3_in", "server_3_out", object=server3)
    topology.add_edge("server_4_in", "server_4_out", object=server4)

    # --- Create Lists of Servers and Communication Links ---
    servers = [server1, server2, server3, server4]
    communication_links = [
        link_c1_s1, link_c1_s2, link_s1_s3, link_s1_s4,
        link_s2_s3, link_s2_s4, link_s3_sink, link_s4_sink
    ]


    #Give client Topology
    # num_stages=2 indicates that the job will pass through 2 "stages" of servers.
    client.add_topology(topology)
    # Create swarms for each stage
    swarm_0 = Swarm(swarm_id=0, servers=[], clients=[client], comm_links=[link_c1_s1, link_c1_s2])
    swarm_stage_1 = Swarm(swarm_id=1, servers=[server1, server2], clients=[], comm_links=[link_s1_s3, link_s1_s4, link_s2_s3, link_s2_s4])
    swarm_stage_2 = Swarm(swarm_id=2, servers=[server3, server4], clients=[], comm_links=[link_s3_sink, link_s4_sink])

    swarms = [swarm_0, swarm_stage_1, swarm_stage_2]
    client.add_swarms(swarms)
    current_time = 0.1
    # Determine which routing policy to use based on the config
    with open('test_config.json', 'r') as f:
        config = json.load(f)
    
    routing_policy = config.get("routing_policy", "round_robin")

    if routing_policy == "min_cost_max_flow":
        #test_min_cost_max(client, servers, communication_links, current_time)
        test_min_cost_max_B(swarms, client, current_time)

    else:
        test_round_robin_routing(client, servers, communication_links, current_time)

def test_round_robin_routing(client, servers, communication_links, current_time):
    # ============================================================
    # 1. TEST round_robin routing
    # ============================================================
    print("\n[TEST] round_robin routing")
    # Unpack communication links and servers
    link_c1_s1, link_c1_s2, link_s1_s3, link_s1_s4, link_s2_s3, link_s2_s4, link_s3_sink, link_s4_sink = communication_links
    server1, server2, server3, server4 = servers


    # Client-controlled job launch
    job_id_rr = client.start_new_job(current_time)
    print(f"Client started Job ID (round_robin): {job_id_rr}")
    client.launch_job_stop()
    # Simulate time steps
    for _ in range(50):
        link_s4_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s3_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        server4.run_time_step(time_quantum=0.1, current_time=current_time)
        server3.run_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        server2.run_time_step(time_quantum=0.1, current_time=current_time)
        server1.run_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s2.process_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s1.process_time_step(time_quantum=0.1, current_time=current_time)
        client.process_time_step(time_quantum=0.1, current_time=current_time)
        current_time += 0.1

    print("\nRouting tests completed.")        

def test_min_cost_max(client, servers, communication_links, current_time):
    # ============================================================
    # 2. TEST min_cost_max_flow routing
    # ============================================================
    print("\n[TEST] min_cost_max_flow routing")

    link_c1_s1, link_c1_s2, link_s1_s3, link_s1_s4, link_s2_s3, link_s2_s4, link_s3_sink, link_s4_sink = communication_links
    server1, server2, server3, server4 = servers

    # Client-controlled job launch
    job_id_mcf = client.start_new_job(current_time)

    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf}")
    job_id_mcf2 = client.start_new_job(current_time)
    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf2}")
    #client.launch_job_stop()


    # Simulate time steps
    for _ in range(2000):
        link_s4_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s3_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        server4.run_time_step(time_quantum=0.1, current_time=current_time)
        server3.run_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        server2.run_time_step(time_quantum=0.1, current_time=current_time)
        server1.run_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s2.process_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s1.process_time_step(time_quantum=0.1, current_time=current_time)
        client.process_time_step(time_quantum=0.1, current_time=current_time)
        current_time += 0.1
    print("\nRouting tests completed.")    
    job_id_mcf2 = client.start_new_job(current_time)
    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf2}")

    for _ in range(2000):
        link_s4_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s3_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        server4.run_time_step(time_quantum=0.1, current_time=current_time)
        server3.run_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        server2.run_time_step(time_quantum=0.1, current_time=current_time)
        server1.run_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s2.process_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s1.process_time_step(time_quantum=0.1, current_time=current_time)
        client.process_time_step(time_quantum=0.1, current_time=current_time)
        current_time += 0.1
    print("\nRouting tests completed.")    
    job_id_mcf3 = client.start_new_job(current_time)
    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf3}")
    for _ in range(2000):
        link_s4_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s3_sink.process_time_step(time_quantum=0.1, current_time=current_time)
        server4.run_time_step(time_quantum=0.1, current_time=current_time)
        server3.run_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s2_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s4.process_time_step(time_quantum=0.1, current_time=current_time)
        link_s1_s3.process_time_step(time_quantum=0.1, current_time=current_time)
        server2.run_time_step(time_quantum=0.1, current_time=current_time)
        server1.run_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s2.process_time_step(time_quantum=0.1, current_time=current_time)
        link_c1_s1.process_time_step(time_quantum=0.1, current_time=current_time)
        client.process_time_step(time_quantum=0.1, current_time=current_time)
        current_time += 0.1
    print("\nRouting tests completed.")    

def test_min_cost_max_B(swarms, client, current_time):
    # ============================================================
    # 2. TEST min_cost_max_flow routing
    # ============================================================
    print("\n[TEST] min_cost_max_flow routing")

    i =0

    # Client-controlled job launch
    job_id_mcf = client.start_new_job(current_time)

    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf}")
    job_id_mcf2 = client.start_new_job(current_time)
    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf2}")
    #client.launch_job_stop()


    # Simulate time steps
    for _ in range(2000):
        for swarm in reversed(swarms):
            swarm.process_time_step(time_quantum=0.1, current_time=current_time)
        current_time += 0.1
    print("\nRouting tests completed.")    
    job_id_mcf2 = client.start_new_job(current_time)
    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf2}")

    for _ in range(2000):
        for swarm in reversed(swarms):
            swarm.process_time_step(time_quantum=0.1, current_time=current_time)

        current_time += 0.1
    print("\nRouting tests completed.")    
    job_id_mcf3 = client.start_new_job(current_time)
    print(f"Client started Job ID (min_cost_max_flow): {job_id_mcf3}")
    for _ in range(2000):
        for swarm in reversed(swarms):
            swarm.process_time_step(time_quantum=0.1, current_time=current_time)

        current_time += 0.1
    print("\nRouting tests completed.")   

def test_externally_launched_job(client):
    # ============================================================
    # 3. TEST externally launched job
    # ============================================================
    print("\n[TEST] External job launch simulation")

    # This part depends on how your system expects external jobs.
    # If you don't have a dedicated method, you might do something like:
    if hasattr(client, "launch_job_from_outside"):
        # Example: pass a job_id or some unique identifier
        external_job_id = client.launch_job_from_outside(job_id=999, arrival_time=current_time)
        print(f"Externally launched job with ID: {external_job_id}")
    else:
        # Fallback: if no dedicated method, demonstrate adding job to queue:
        print("No 'launch_job_from_outside' method found; simulating manual enqueue.")
        # You may need to adapt this line to match your actual queue usage
        client.jobs_queue.append({
            'job_id': 999,
            'start_time': current_time,
            'remaining_stages': client.num_stages
        })

    # Simulate time steps
    for _ in range(5):
        client.process_time_step(current_time, time_quantum=1.0)
        current_time += 1

    print("\nAll routing and job launch tests completed.")

if __name__ == "__main__":
    inizilatazion_of_objects()
    os.remove('test_config.json')