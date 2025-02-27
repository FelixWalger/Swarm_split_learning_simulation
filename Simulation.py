import os
import random
import sys
import copy
import json
import matplotlib.pyplot as plt
import datetime

from System import System
from Server import Server
from Communicationlink import CommunicationLink
from DecoderBlock import DecoderBlock
from Client import Client
import csv

class Simulation:
    def __init__(self, config_filename='config_simulation.json', num_servers=5, num_clients=2, num_decoder_blocks=6, num_swarms=2, steps=10000):
        time_quantum = 0.1
        global_steps = 1000
        config_path = os.path.join(os.path.dirname(__file__), config_filename)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_filename} not found.")
        with open(config_path, 'r') as config_file:
            config_simulation = json.load(config_file)
            num_servers = config_simulation.get('num_servers', num_servers)
            num_clients = config_simulation.get('num_clients', num_clients)
            num_decoder_blocks = config_simulation.get('num_decoder_blocks', num_decoder_blocks)
            num_swarms = config_simulation.get('num_swarms', num_swarms)
            steps = config_simulation.get('local_steps', steps)
            time_quantum = config_simulation.get('time_quantum', 0.1)
            global_steps = config_simulation.get('global_steps', 1000)
          
        self.system_id_counter = 0



        self.num_servers = num_servers
        self.num_clients = num_clients
        self.num_decoder_blocks = num_decoder_blocks
        self.num_swarms = num_swarms
        self.steps = steps
        self.time_quantum = time_quantum
        self.global_steps = global_steps
        self.servers = []
        self.clients = []
        self.comm_links = []
        self.decoder_blocks = []
        self.system = None
        self.systems = []

    def setup_servers(self):
        for i in range(self.num_servers):
            srv = Server(server_id=i)
            self.servers.append(srv)

    def setup_communication_links(self):
        link_id = 0
        for i in range(len(self.servers)):
            for j in range(len(self.servers)):
                if i != j:
                    link = CommunicationLink(
                        link_id=link_id,
                        from_entity=self.servers[i],
                        to_entity=self.servers[j],
                    )
                    self.comm_links.append(link)
                    link_id += 1

        for server in self.servers:
            for client in self.clients:
                link = CommunicationLink(
                    link_id=link_id,
                    from_entity=server,
                    to_entity=client,
                )
                self.comm_links.append(link)
                link_id += 1

        for client in self.clients:
            for server in self.servers:
                link = CommunicationLink(
                    link_id=link_id,
                    from_entity=client,
                    to_entity=server,
                )
                self.comm_links.append(link)
                link_id += 1

    def setup_clients(self):
        for i in range(1, self.num_clients + 1):
            client = Client(client_id=i, num_stages=self.num_swarms)
            self.clients.append(client)

    def setup_decoder_blocks(self):
        for block_id in range(self.num_decoder_blocks):
            block = DecoderBlock(block_id=block_id, config_filename='config_simulation.json')
            self.decoder_blocks.append(block)

    def setup_system(self):
        self.system = System(
            system_id=self.system_id_counter,
            system_init="server_commlink_object_based",
            servers_list=self.servers.copy(),
            commlinks_list=self.comm_links.copy(),
            clients_list=self.clients.copy(),
            DecoderBlock_list=self.decoder_blocks,
            config_filename='config_simulation.json'
        )
        self.system_id_counter += 1

    def deepcopy_comm_links(self, comm_links):
        comm_links_copy = []
        for link in comm_links:
            comm_links_copy.append(copy.copy(link))
        return comm_links_copy    
    
    def get_servers_and_client_from_comm_links(self, comm_links):
        servers = []
        clients = []
        for link in comm_links:
            if isinstance(link.from_entity, Server) and link.from_entity not in servers:
                servers.append(link.from_entity)
            elif isinstance(link.from_entity, Client) and link.from_entity not in clients:
                clients.append(link.from_entity)
        return servers, clients
    
    def replace_servers_and_clients_at_comm_links(self, comm_links, servers, clients):
        server_copies = [copy.copy(server) for server in servers]
        client_copies = [copy.copy(client) for client in clients]

        for server in server_copies:
            server.incoming_queues = {}  # {job_id[job_obj]:job_obj,  weight)}
            server.outgoing_queues = {}  # {job_id: [job_obj]}
            server.job_metadata = {}     # Metadata for each job (time tracking, weights, etc.)
            server.current_open_jobs = []
            server.completed_jobs_log = []
            server.memory_usage_tracking = {}
            server.decoder_blocks = {} 
            server.num_blocks = 0
            server.lowest_decoder_block_id = None

            # Memory tracking initialization.
            server.kv_mem_used = 0
            server.token_buffer_used = 0
            server.model_param_mem_used = 0
            server.kv_mem_allocated = 0
            server.token_buffer_allocated = 0
            server.model_param_mem_allocated = 0
            server.all_memory_used = 0
            server.all_memory_allocated = 0

        for client in client_copies:
            client.outgoing_job_iterations = []
            client.job_metadata = {} 
            client.round_robin_dict = {}
            client.incoming_job_iterations = []  
            client.routing_history = []     
        # A global iteration counter to

        for link in comm_links:
            if isinstance(link.from_entity, Server):
                link.from_entity = server_copies[link.from_entity.server_id]
            elif isinstance(link.from_entity, Client):
                link.from_entity = client_copies[link.from_entity.client_id - 1]

            if isinstance(link.to_entity, Server):
                link.to_entity = server_copies[link.to_entity.server_id]
            elif isinstance(link.to_entity, Client):
                link.to_entity = client_copies[link.to_entity.client_id - 1]

        return comm_links, server_copies, client_copies
    
    def change_servers_selection_strategy(self, servers, selection_strategy):
        for server in servers:
            server.selection_strategy = selection_strategy

    def change_clients_routing_policy(self, clients, routing_policy):
        for client in clients:
            client.routing_policy = routing_policy
           

    def setup_system_clone_with_change(self, topology_policy, routing_policy, selection_strategy):    
        comm_links_copy = self.deepcopy_comm_links(self.system.communication_links)
        servers, clients = self.get_servers_and_client_from_comm_links(comm_links_copy)
        comm_links_copy, servers, clients = self.replace_servers_and_clients_at_comm_links(comm_links_copy, servers, clients)
        self.change_servers_selection_strategy(servers, selection_strategy)
        
        self.change_clients_routing_policy(clients, routing_policy)

        self.clients.extend(clients)
        self.servers.extend(servers)
        self.comm_links.extend(comm_links_copy)
        system = System(
            system_id=self.system_id_counter,
            system_init="server_commlink_object_based",
            servers_list=servers,
            commlinks_list=comm_links_copy,
            topology_policy=topology_policy,
            routing_policy=routing_policy,
            selection_strategy_at_servers=selection_strategy,
            clients_list=clients,
            DecoderBlock_list=self.decoder_blocks,
            config_filename='config_simulation.json'
        )
        self.system_id_counter += 1
        return system
    
    def run_time_steps_on_all_systems(self):
        time_quantum=self.time_quantum 
        central_steps = self.global_steps
        steps = self.steps
        start_time=0.1
        last_time = 0.1
        for central_step in range(central_steps):
            for system in self.systems:
                last_time = system.run_time_steps(time_quantum=time_quantum, start_time=start_time, steps=steps)
            if central_step == central_steps - 4:
                for client in self.clients:
                    client.launch_job_stop()
            start_time = last_time

    
    def final_analyzes_of_job_work_periods(self):
        for system in self.systems:
            total_time = 0
            total_jobs = 0
            for client in system.clients:
                for job_id, job_data in client.job_metadata.items():
                    if job_data["end_time"] is not None:
                        total_time += job_data["end_time"] - job_data["start_time"]
                        total_jobs += 1
            if total_jobs > 0:
                average_job_time = total_time / total_jobs
                if system.clients and system.servers:
                    client = system.clients[0]
                    server = system.servers[0]
                    topology_policy = system.topology_policy
                    routing_policy = client.routing_policy
                    selection_strategy = server.selection_strategy
                    print(f"System ID {system.system_id} - Average job time: {average_job_time}, Topology Policy: {topology_policy}, Routing Policy: {routing_policy}, Selection Strategy: {selection_strategy}")
                else:
                    print(f"System ID {system.system_id} - Average job time: {average_job_time}")
                print(f"System ID {system.system_id} - Average job time: {average_job_time}")
            else:
                print(f"System ID {system.system_id} - No completed jobs to analyze.")


    def plot_results(self):
        all_total_tokens = []
        all_average_job_time = []
        for system in self.systems:
            total_tokens_list = []
            average_job_time_list = []
            job_data_dict = {}
            for client in system.clients:
                for job_id, job_data in client.job_metadata.items():
                    if job_data["end_time"] is not None:
                        job_length = job_data["total_tokens"]
                        job_time = job_data["end_time"] - job_data["start_time"]
                        if job_length not in job_data_dict:
                            job_data_dict[job_length] = []
                        job_data_dict[job_length].append(job_time)

            sorted_job_lengths = sorted(job_data_dict.keys())
            for job_length in sorted_job_lengths:
                job_times = job_data_dict[job_length]
                average_job_time = sum(job_times) / len(job_times)
                total_tokens_list.append(job_length)
                average_job_time_list.append(average_job_time)
            plt.figure()
            plt.plot(total_tokens_list, average_job_time_list, 'o-')
            plt.xlabel('Total Tokens [tok]')
            plt.ylabel('Average Job Time [sec]')
            plt.title(f'System ID {system.system_id} - Average Job Time vs Total Tokens')
            plt.grid(True)
            if system.clients and system.servers:
                client = system.clients[0]
                server = system.servers[0]
                topology_policy = system.topology_policy
                routing_policy = client.routing_policy
                selection_strategy = server.selection_strategy
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"plot_{current_time}_topology_{topology_policy}_routing_{routing_policy}_selection_{selection_strategy}.png"
                plt.savefig(filename)
            plt.show()

            # Unscaled version
            plt.figure()
            plt.plot(total_tokens_list, average_job_time_list, 'o-')
            plt.xlabel('Total Tokens [tok]')
            plt.ylabel('Average Job Time [sec]')
            plt.title(f'System ID {system.system_id} - Average Job Time vs Total Tokens (Unscaled)')
            plt.grid(True)
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            if system.clients and system.servers:
                client = system.clients[0]
                server = system.servers[0]
                topology_policy = system.topology_policy
                routing_policy = client.routing_policy
                selection_strategy = server.selection_strategy
                current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"plot_{current_time}_topology_{topology_policy}_routing_{routing_policy}_selection_{selection_strategy}_B.png"
                plt.savefig(filename)
            plt.show()
            all_total_tokens.append(total_tokens_list)
            all_average_job_time.append(average_job_time_list)
            plt.figure()

        for total_tokens_list, average_job_time_list in zip(all_total_tokens, all_average_job_time):
            plt.plot(total_tokens_list, average_job_time_list, 'o-')
        plt.xlabel('Total Tokens [tok]')
        plt.ylabel('Average Job Time [sec]')
        plt.title('Average Job Time vs Total Tokens for all Systems')
        plt.grid(True)
        plt.legend([f"System {system.system_id}: {system.topology_policy}, {system.clients[0].routing_policy}, {system.servers[0].selection_strategy}" for system in self.systems], loc='best')
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{current_time}_all_system_configs.png"
        plt.savefig(filename)
        plt.show()

        # Unscaled version
        plt.figure()
        for total_tokens_list, average_job_time_list in zip(all_total_tokens, all_average_job_time):
            plt.plot(total_tokens_list, average_job_time_list, 'o-')
        plt.xlabel('Total Tokens [tok]')
        plt.ylabel('Average Job Time [sec]')
        plt.title('Average Job Time vs Total Tokens for all Systems (Unscaled)')
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend([f"System {system.system_id}: {system.topology_policy}, {system.clients[0].routing_policy}, {system.servers[0].selection_strategy}" for system in self.systems], loc='best')
        filename = f"plot_{current_time}_all_system_configs_B.png"
        plt.savefig(filename)
        plt.show()

    def handling_results(self):       
        for system in self.systems:
            job_data_dict = {}
            for client in system.clients:
                for job_id, job_data in client.job_metadata.items():
                    if job_data["end_time"] is not None:
                        job_length = job_data["total_tokens"]
                        job_time = job_data["end_time"] - job_data["start_time"]
                        if job_length not in job_data_dict:
                            job_data_dict[job_length] = []
                        job_data_dict[job_length].append(job_time)

            # Create CSV file
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{current_time}_system_{system.system_id}_global_steps_{self.global_steps}_local_steps_{self.steps}_{system.topology_policy}_{system.clients[0].routing_policy}_{system.servers[0].selection_strategy}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Job Length', 'Job Times', 'Topology Policy', 'Routing Policy', 'Selection Strategy', 'Num Servers', 'Num Clients', 'Num Swarms', 'Global Steps', 'Local Steps'])
                for job_length, job_times in job_data_dict.items():
                    writer.writerow([
                        job_length, 
                        job_times, 
                        f"topology: {system.topology_policy}", 
                        f"routing: {system.clients[0].routing_policy}", 
                        f"selection: {system.servers[0].selection_strategy}", 
                        f"num_servers: {self.num_servers}", 
                        f"num_clients: {self.num_clients}", 
                        f"num_decoder_blocks: {self.num_decoder_blocks}",
                        f"num_swarms: {self.num_swarms}", 
                        f"global_steps: {self.global_steps}", 
                        f"local_steps: {self.steps}"
                    ])

    def read_and_plot_results(self, csv_file_paths):
        all_total_tokens = []
        all_average_job_time = []

        for file_path in csv_file_paths:
            job_data_dict = {}
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    job_length = int(row[0])
                    job_times = json.loads(row[1])
                    job_data_dict[job_length] = job_times

            total_tokens_list = sorted(job_data_dict.keys())
            average_job_time_list = [sum(job_data_dict[job_length]) / len(job_data_dict[job_length]) for job_length in total_tokens_list]

            all_total_tokens.append(total_tokens_list)
            all_average_job_time.append(average_job_time_list)

            plt.figure()
            plt.plot(total_tokens_list, average_job_time_list, 'o-')
            plt.xlabel('Total Tokens [tok]')
            plt.ylabel('Average Job Time [sec]')
            plt.title(f'Average Job Time vs Total Tokens for {os.path.basename(file_path)}')
            plt.grid(True)
            plt.show()

        # Plot all results together
        plt.figure()
        for total_tokens_list, average_job_time_list in zip(all_total_tokens, all_average_job_time):
            plt.plot(total_tokens_list, average_job_time_list, 'o-')
        plt.xlabel('Total Tokens [tok]')
        plt.ylabel('Average Job Time [sec]')
        plt.title('Average Job Time vs Total Tokens for all CSV files')
        plt.grid(True)
        plt.legend([os.path.basename(file_path) for file_path in csv_file_paths], loc='best')
        plt.show()




    def run(self):
        random.seed(42)
        self.setup_servers()
        self.setup_clients()
        self.setup_communication_links()
        self.setup_decoder_blocks()
        self.setup_system()
        self.systems.append(self.system)


        system = self.setup_system_clone_with_change(topology_policy="latency", routing_policy="min_cost_max_flow", selection_strategy="FIFO")
        self.systems.append(system)
        system = self.setup_system_clone_with_change(topology_policy="latency", routing_policy="round_robin", selection_strategy="longest_time_not_seen")
        self.systems.append(system)
        system = self.setup_system_clone_with_change(topology_policy="latency", routing_policy="round_robin", selection_strategy="FIFO")
        self.systems.append(system)
        system = self.setup_system_clone_with_change(topology_policy="throughput", routing_policy="min_cost_max_flow", selection_strategy="longest_time_not_seen")
        self.systems.append(system)
        system = self.setup_system_clone_with_change(topology_policy="throughput", routing_policy="min_cost_max_flow", selection_strategy="FIFO")
        self.systems.append(system)
        system = self.setup_system_clone_with_change(topology_policy="throughput", routing_policy="round_robin", selection_strategy="longest_time_not_seen")
        self.systems.append(system)
        system = self.setup_system_clone_with_change(topology_policy="throughput", routing_policy="round_robin", selection_strategy="FIFO")
        self.systems.append(system)

        for system in self.systems:
            system.make_system_init()
       
        self.run_time_steps_on_all_systems()
        self.final_analyzes_of_job_work_periods()
        self.handling_results()
        self.plot_results()
        print("----- Finished  Swarm Creation -----")

def main():
    simulation = Simulation()
    simulation.run()

if __name__ == "__main__":
    main()
