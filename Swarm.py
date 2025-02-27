import random
import numpy
from Communicationlink import CommunicationLink
from Server import Server
from Client import Client

class Swarm:
    """
    A Swarm manages a group of servers and their outgoing communication links.
    Responsibilities:
      1) At the start of each time step, look at each server's outgoing queues.
      2) For each waiting job iteration, try to push it onto the corresponding
         communication link (if that link is functioning).
      3) Then let each link process its own time step (transmissions, etc.).
      4) Allows servers or links to be added/removed (swarm reorganization).
    """

    def __init__(self, swarm_id, servers=None, clients = None, comm_links=None):
        """
        :param swarm_id:    Unique identifier for the swarm.
        :param servers:     Optional initial list of servers belonging to this swarm.
        :param comm_links:  Optional initial list of communication links managed by this swarm.
        """
        self.swarm_id = swarm_id
        self.servers = servers if servers else []
        self.comm_links = comm_links if comm_links else []
        self.clients = clients if clients else []

    def add_server(self, server):
        """
        Add a server to this swarm.
        """
        if server not in self.servers:
            self.servers.append(server)

    def remove_server(self, server):
        """
        Remove a server from this swarm.
        """
        if server in self.servers:
            self.servers.remove(server)

    def add_comm_link(self, link):
        """
        Add a communication link to this swarm.
        """
        if link not in self.comm_links:
            self.comm_links.append(link)

    def remove_comm_link(self, link):
        """
        Remove a communication link from this swarm.
        """
        if link in self.comm_links:
            self.comm_links.remove(link)

    def add_client(self, client):
        """
        Add a client to this swarm.
        """
        if client not in self.clients:
            self.clients.append(client)

    def remove_client(self, client):
        """
        Remove a client from this swarm.
        """
        if client in self.clients:
            self.clients.remove(client)        


    def collect_throughput_info_server(self):
        """
        Collect throughput information from all servers in the swarm.
        """
        throughput_info = {}
        for server in self.servers:
            throughput_info[server.id] = server.get_throughput_info()
        return throughput_info

    def calculate_average_throughput_servers(self):
        """
        Calculate the average throughput across all servers in the swarm.
        """
        total_throughput = 0
        num_servers = len(self.servers)
        for server in self.servers:
            total_throughput += server.get_throughput_info()["actual"]
        return total_throughput / num_servers if num_servers > 0 else 0
    
    def calucate_average_latency_servers(self):
        """
        Calculate the average latency across all servers in the swarm.
        """
        if not self.servers and self.clients:
            return 0
        return 1/self.calculate_average_throughput_servers() if self.calculate_average_throughput_servers() > 0 else 100000000000
    
    def calculate_avergae_latencies_comm_links(self):
        """
        Calculate the average latency across all communication links in the swarm.
        """
        total_latency = 0
        num_links = len(self.comm_links)
        for link in self.comm_links:
            total_latency += link.get_latency()
        return total_latency / num_links if num_links > 0 else 0

    def count_decoder_blocks(self):
        """
        Count the total number of decoder blocks held by all servers in the swarm.
        """
        if not self.servers:
            return 0
        # Assuming all servers have the same decoder blocks
        return len(self.servers[0].decoder_blocks)


    def process_time_step(self, time_quantum, current_time):
        """
        1) For each server in the swarm, check its outgoing queues.
           - For each waiting job iteration, see if the designated link is functioning.
           - If it is up, move the job iteration to the link via add_job_iteration.
           - Otherwise, leave it in the queue for a future attempt.
        2) Then call process_time_step on each communication link to
           advance transmissions (fair-sharing, propagation, etc.).
        """
                # Step 2: Let each link process transmissions for this time step
        for link in self.comm_links:
            link.process_time_step(time_quantum=time_quantum, current_time=current_time)

        # Step 1: Move job iterations from servers -> comm links (if link is up)
        for server in self.servers:
           server.run_time_step(time_quantum=time_quantum, current_time=current_time)

        for client in self.clients:
            client.process_time_step(time_quantum=time_quantum, current_time=current_time)   




    @classmethod
    def from_existing_swarm(cls, existing_swarm, config_filename=None):
        """
        Create a new Swarm instance from an existing swarm.
        
        :param existing_swarm: The existing Swarm instance to copy from.
        :param config_filename: Optional configuration filename.
        :return: A new Swarm instance.
        """
        new_swarm = cls(
            swarm_id=existing_swarm.swarm_id,
            servers=[Server.from_existing_server(server, config_filename=config_filename) for server in existing_swarm.servers],
            comm_links=[CommunicationLink.from_existing_comm_link(link, config_filename=config_filename) for link in existing_swarm.comm_links]
        )
        new_swarm.config_filename = config_filename
        return new_swarm