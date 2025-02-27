import os
import json
from Server import Server
from Communicationlink import CommunicationLink
from Swarm import Swarm
from Client import Client

def create_test_config():
    config = {
        "server_properties": {
            "throughput_range": [50, 150],
            "memory_range": [1024, 4096],
            "decoder_block_memory_range": [128, 256],
            "d_model_memory_range": [256, 512],
            "KV_cache_range": [256, 512],
            "dropout_prob_range": [0.01, 0.1],
            "selection_strategy": ["FIFO", "longest_time_not_seen"],
            "down_server_time_range_min": [1, 5],
            "up_server_time_range_min": [1, 5],
            "degradation_rate": [0.01, 0.1]
        },
        "network_properties": {
            "data_rate_range_Mbits/s": [10, 100],
            "propagation_speed_km/s": 200,
            "distance_range_km": [1, 10],
            "jitter_lognorm_param_tuples_mu_sigma": [(0.1, 0.01), (0.2, 0.02)],
            "down_link_time_range_min": [1, 5],
            "up_link_time_range_min": [1, 5]
        },
        "client_properties": {
            "max_token_length_range": [50, 100]
        },
        "precision": 32,
        "d_model": 512
    }
    with open('test_config.json', 'w') as f:
        json.dump(config, f)

def test_from_existing_server():
    create_test_config()
    original_server = Server(server_id=1, config_filename='test_config.json')
    new_server = Server.from_existing_server(original_server, config_filename='test_config.json')
    assert original_server.id == new_server.id
    assert original_server.memory_capacity == new_server.memory_capacity
    assert original_server.throughput_dist_std_factor == new_server.throughput_dist_std_factor
    assert original_server.base_throughput == new_server.base_throughput
    assert original_server.degrading_factor == new_server.degrading_factor
    assert original_server.decoder_blocks.keys() == new_server.decoder_blocks.keys()
    print("Server test passed.")

def test_from_existing_comm_link():
    create_test_config()
    original_link = CommunicationLink(link_id=1, from_entity="A", to_entity="B", config_filename='test_config.json')
    new_link = CommunicationLink.from_existing_link(original_link, config_filename='test_config.json')
    assert original_link.link_id == new_link.link_id
    assert original_link.from_entity == new_link.from_entity
    assert original_link.to_entity == new_link.to_entity
    assert original_link.data_rate_bps == new_link.data_rate_bps
    assert original_link.speed_of_signal_m_s == new_link.speed_of_signal_m_s
    assert original_link.physical_length_m == new_link.physical_length_m
    print("CommunicationLink test passed.")

def test_from_existing_client():
    create_test_config()
    topology = nx.DiGraph()
    original_client = Client(client_id=1, server_commLink_Topology=topology, config_filename='test_config.json')
    new_client = Client.from_existing_client(original_client, client_id=2)
    assert original_client.id != new_client.id
    assert original_client.max_token_length == new_client.max_token_length
    assert original_client.min_token_length == new_client.min_token_length
    assert original_client.job_launch_probability == new_client.job_launch_probability
    assert original_client.routing_policy == new_client.routing_policy
    print("Client test passed.")

def test_from_existing_swarm():
    create_test_config()
    server1 = Server(server_id=1, config_filename='test_config.json')
    server2 = Server(server_id=2, config_filename='test_config.json')
    link1 = CommunicationLink(link_id=1, from_entity="A", to_entity="B", config_filename='test_config.json')
    link2 = CommunicationLink(link_id=2, from_entity="B", to_entity="C", config_filename='test_config.json')
    client1 = Client(client_id=1, server_commLink_Topology=nx.DiGraph(), config_filename='test_config.json')
    client2 = Client(client_id=2, server_commLink_Topology=nx.DiGraph(), config_filename='test_config.json')
    original_swarm = Swarm(swarm_id=1, servers=[server1, server2], comm_links=[link1, link2])
    new_swarm = Swarm.from_existing_swarm(original_swarm, config_filename='test_config.json')
    assert original_swarm.swarm_id == new_swarm.swarm_id
    assert len(original_swarm.servers) == len(new_swarm.servers)
    assert len(original_swarm.comm_links) == len(new_swarm.comm_links)
    print("Swarm test passed.")

if __name__ == "__main__":
    test_from_existing_server()
    test_from_existing_comm_link()
    test_from_existing_client()
    test_from_existing_swarm()
    os.remove('test_config.json')