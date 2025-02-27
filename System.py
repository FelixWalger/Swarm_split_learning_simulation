import random
import json
import numpy as np
import pulp
import networkx as nx
import copy
import os

from Communicationlink import CommunicationLink
from Server import Server
from Client import Client
from Swarm import Swarm
from DecoderBlock import DecoderBlock
from MIP_opti import OptiProblem
import os

class System:
    def __init__(
        self,
        system_id,
        system_init,

        num_clients=None,
        num_servers=None,

        servers_list=None,
        clients_list=None,

        DecoderBlock_list=None,
        commlinks_list=None,
        swarms_list=None,                 
          # <--- NEW: optional pre-built swarms
        max_token_length=None,
        routing_policy=None,
        topology_policy=None,
        selection_strategy_at_servers=None,
        topology_dict=None,
        config_filename='config_simulation.json'
    ):
        """
        :param num_clients: Number of clients to create (if not provided externally).
        :param num_servers: Number of servers to create (if not provided externally).
        :param servers_list: Pre-built list of Server objects (optional).
        :param clients_list: Pre-built list of Client objects (optional).
        :param commlinks_list: Pre-built list of CommunicationLink objects (optional).
        :param swarms: If provided, we skip building servers/links randomly and use these swarms
                   to gather servers, clients, and links. Each Swarm has .servers, .clients, .comm_links.
        :param max_token_length: Maximum token length used for clients.
        :param routing_policy: e.g. "round_robin" or "min_cost_max_flow".
        :param topology_policy: Placeholder for how we build or interpret the topology.
        :param selection_strategy_at_servers: e.g. "FIFO", "longest_time_not_seen", etc.
        :param config_filename: JSON config with extra simulation parameters.
        """
        config_path = os.path.join(os.path.dirname(__file__), config_filename)
        with open(config_path, 'r') as config_file:
            config_simulation = json.load(config_file)

        self.max_token_length = max_token_length if max_token_length is not None else config_simulation['system_properties']['max_sequence_length'] if 'system_properties' in config_simulation and 'max_sequence_length' in config_simulation['system_properties'] else None
        self.routing_policy = routing_policy if routing_policy is not None else config_simulation['routing_policy'] if 'routing_policy' in config_simulation else None
        self.topology_policy = topology_policy if topology_policy is not None else config_simulation['topology_policy'] if 'topology_policy' in config_simulation else None
        self.selection_strategy_at_servers = selection_strategy_at_servers if selection_strategy_at_servers is not None else config_simulation['server_properties']['selection_strategy'] if 'server_properties' in config_simulation and 'selection_strategy' in config_simulation['server_properties'] else None

        self.num_swarms = len(swarms_list) if swarms_list else config_simulation['num_swarms'] if 'num_swarms' in config_simulation else 2
        self.swarms = swarms_list  

        self.system_id = system_id
        self.system_init = system_init
        self.d_model = config_simulation['d_model'] if 'd_model' in config_simulation else 512
        self.precision = config_simulation['precision'] if 'precision' in config_simulation else 16
        
        self.num_decoder_blocks = len(DecoderBlock_list) if DecoderBlock_list else config_simulation['num_Decoder_Blocks'] if 'num_Decoder_Blocks' in config_simulation else 2
        self.DecoderBlock_list = DecoderBlock_list

        self.num_clients = len(clients_list) if clients_list else config_simulation['num_clients'] if 'num_clients' in config_simulation else None
        self.num_servers = len(servers_list) if servers_list else config_simulation['num_servers'] if 'num_servers' in config_simulation else None

        # Could be None or a list of Swarm objects
        # The final sets of servers/clients/links in the entire system:
        self.servers = []
        self.clients = []
        self.communication_links = []
        self.reduced_communication_links = []

        # Initialize from:
        #  (A) swarms, if provided
        #  (B) or user-supplied lists
        #  (C) or newly created random ones
        if self.swarms:
            # Gather all servers/clients/links from the swarms
            self._init_from_swarms()
        else:
            # Use the explicit lists or fallback to random creation
            self.servers = servers_list if servers_list is not None else []
            self.clients = clients_list if clients_list is not None else []
            self.communication_links = commlinks_list if commlinks_list is not None else []


        # Possibly load config
        self.config_filename = config_filename
        self.config = self._load_config(self.config_filename)

        # Build or store a topology graph if needed
        # (If you want an NX graph from swarms or from servers/links, do it here)
        self.topology_dict = topology_dict
        self.topology = None
        self.swarms = []

        # Provide each client with references to the swarms or the full system (optional)


    def make_system_init(self):
        """
        Placeholder for initializing the system.
        This is a no-op here, just demonstrating the idea.
        """
        if self.system_init == "server_commlink_object_based":
            if self.DecoderBlock_list is None:
                self._init_decoder_blocks()
            self.topology_dict = self.create_topology(self.topology_dict)
            G = self._create_topology_graph()
            for client in self.clients:
                # Make a deep copy of the original graph
                G_copy = G.copy()
                
                # List all incoming edges to client_{client.id}, along with any edge data
                incoming_edges = list(G_copy.in_edges(f"client_{client.id}", data=True))
                
                # For each incoming edge, remove it, then add an edge to "SINK_in"
                for src, tgt, edata in incoming_edges:
                    G_copy.remove_edge(src, tgt)
                    if 'object' in edata:
                        G_copy.add_edge(src, "SINK_in", object=edata['object'])
                    else:
                        G_copy.add_edge(src, "SINK_in")
                # Remove other clients from the topology
                for other_client in self.clients:
                    if other_client.id != client.id:
                        G_copy.remove_node(f"client_{other_client.id}")
                # IMPORTANT: make sure to pass the modified graph
                client.add_topology(G_copy)
            
        elif self.system_init == "server_commlink_num_based":
            raise NotImplementedError("server_commlink_num_based initialization is not supported so far.")
            pass
        elif self.system_init == "full_number_based":
            self._init_entities()
            self._init_comm_links()
            self._init_decoder_blocks()
        elif self.system_init == "swarm_based":
            self._init_from_swarms()    
   
    def _assign_decoder_blocks_to_servers(self, server_swarm_dict):
        """
        Assign decoder blocks to servers in a round-robin fashion.
        """
        num_total_blocks = 0
        for sw_id, swarm_info in server_swarm_dict.items():
            if sw_id == 0:
                continue
            num_decoder_blocks = swarm_info["num_decoder_blocks"]
            
            for srv_id in swarm_info["server_ids"]:
                for srv in self.servers:
                    if srv.server_id == srv_id:
                        if self.DecoderBlock_list and self.num_decoder_blocks == len(self.DecoderBlock_list):
                            for block in self.DecoderBlock_list:
                                if num_total_blocks <= block.block_id < num_total_blocks + num_decoder_blocks:
                                    srv.assign_decoder_block(copy.deepcopy(block))
                                    break
                        else:    
                            for block_id in range(num_total_blocks, num_total_blocks + num_decoder_blocks):
                                decoder_block = DecoderBlock(block_id=block_id)
                                srv.assign_decoder_block(decoder_block)
            num_total_blocks += num_decoder_blocks

    def _init_from_swarms(self):
        """
        Gather all servers/clients/communication links from the provided swarms.
        Also store the swarms in a dictionary keyed by their ID if desired.
        """
        self.swarms_dict = {}  # { swarm_id -> Swarm(...) }
        for sw in self.swarms:
            self.swarms_dict[sw.swarm_id] = sw
            # Add servers
            for srv in sw.servers:
                if srv not in self.servers:
                    self.servers.append(srv)
            # Add clients
            for cli in sw.clients:
                if cli not in self.clients:
                    self.clients.append(cli)
            # Add comm links
            for link in sw.comm_links:
                if link not in self.communication_links:
                    self.communication_links.append(link)

    def _load_config(self, filename):
        """
        Simple helper to load the JSON config file, or return an empty dict if not found.
        """
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def _init_entities(self):
        """
        Internal helper to initialize client and server objects if they are not already populated.
        """
        # If we have no clients_list but user said 'num_clients', create them
        if self.num_clients is not None and len(self.clients) < self.num_clients:
            for c_id in range(len(self.clients), self.num_clients):
                client = Client(client_id=c_id)
                self.clients.append(client)

        # If we have no servers_list but user said 'num_servers', create them
        if self.num_servers is not None and len(self.servers) < self.num_servers:
            for s_id in range(len(self.servers), self.num_servers):
                server = Server(server_id=s_id)
                self.servers.append(server)

    def _init_comm_links(self):
        """
        Internal helper to generate random communication links if none exist.
        In a real scenario, you might have a specific deterministic or user-defined topology.
        """
        link_id = 0
        # Connect each client to each server with random latency
        for client in self.clients:
            for server in self.servers:
                latency = random.uniform(1, 5)
                link = CommunicationLink(link_id, client, server)
                self.communication_links.append(link)
                link_id += 1
        # Optionally connect servers among themselves
        for i in range(len(self.servers)):
            for j in range(i+1, len(self.servers)):
                latency = random.uniform(1, 5)
                link = CommunicationLink(link_id, self.servers[i], self.servers[j])
                self.communication_links.append(link)
                link_id += 1

    def _init_decoder_blocks(self):
        """
        Internal helper to initialize decoder blocks if they are not already populated.
        """
        # If we have no decoder_blocks_list but user said 'num_decoder_blocks', create them
        if self.num_decoder_blocks is not None and len(self.DecoderBlock_list) < self.num_decoder_blocks:
            for db_id in range(len(self.DecoderBlock_list), self.num_decoder_blocks):
                decoder_block = DecoderBlock(block_id=db_id)
                self.DecoderBlock_list.append(decoder_block)

    def _create_topology_graph(self):
        """
        If desired, build an NX DiGraph from the known servers/links/clients.
        If you're using a different method to store or use your topology,
        you can skip or adapt this function.
        """
        # Basic example:
        G = nx.DiGraph()

        # For each Server, add server_in -> server_out edge
        for srv in self.servers:
            s_in = f"server_{srv.server_id}_in"
            s_out = f"server_{srv.server_id}_out"
            G.add_node(s_in)
            G.add_node(s_out)
            G.add_edge(s_in, s_out, object=srv)

        # For each Client, just add a node labeled "client_{id}"
        for cli in self.clients:
            c_label = f"client_{cli.id}"
            G.add_node(c_label)

        # For each CommunicationLink, interpret from_entity -> to_entity
        # to the corresponding node labels (server_X_out or server_X_in or client_X).
        for link in self.reduced_communication_links:
            src_label = self._entity_to_label(link.from_entity, suffix="_out")
            dst_label = self._entity_to_label(link.to_entity, suffix="_in")
            G.add_node(src_label)
            G.add_node(dst_label)
            G.add_edge(src_label, dst_label, object=link)

        self.topology = G
        return G

    def _entity_to_label(self, entity, suffix=""):
        """
        Utility to convert an entity (Server, Client, or string) into
        a node label for the NX topology. For servers, use server_{id}_{suffix}.
        For clients, use client_{id}.
        For anything else, just str(entity).
        """
        if isinstance(entity, Server):
            return f"server_{entity.server_id}{suffix}"
        elif isinstance(entity, Client):
            return f"client_{entity.id}"
        else:
            return str(entity)

    def _init_clients_with_swarms(self):
        """
        Optionally provide each client a reference to all swarms or the entire topology.
        This is optional and depends on your usage. 
        """
        # If you want to pass the final swarms list to each client:
        if self.swarms:
            for cli in self.clients:
                if hasattr(cli, "add_swarms"):
                    cli.add_swarms(self.swarms)
        # If you want to pass the built topology:
        if self.topology:
            for cli in self.clients:
                if hasattr(cli, "add_topology"):
                    cli.add_topology(self.topology)

    def create_topology(self, topology_dict=None):
        """
        Placeholder for creating a network topology.
        This is a no-op here, just demonstrating the idea.
        """
        if self.topology_policy == "throughput":
            print("[INFO] Creating topology based on throughput.")
            if topology_dict is None:
                topology_dict = self.create_throughput_balanced_swarms(self.num_swarms, self.num_decoder_blocks)
            self._find_from_topology_dict_comm_link(topology_dict)
            self._find_client_comm_links_from_topology_dict(topology_dict)
            self._create_full_swarms(topology_dict)
            self._assign_decoder_blocks_to_servers(topology_dict)
        elif self.topology_policy == "latency":
            print("[INFO] Creating topology based on latency.")
            if topology_dict is None:
                topology_dict = self.create_latency_balanced_swarms()

            self._find_from_topology_dict_comm_link(topology_dict)
            self._find_client_comm_links_from_topology_dict(topology_dict)
            self._create_full_swarms(topology_dict)
            self._assign_decoder_blocks_to_servers(topology_dict)

        self._create_topology_graph()    
        return topology_dict

    def create_latency_balanced_swarms(self):
        opti_problem = OptiProblem(
            servers=self.servers,
            comm_links=self.communication_links,
            max_L=self.num_swarms,
            num_decoder_blocks=self.num_decoder_blocks,
            SL_max=self.max_token_length,
            d_model=self.d_model,
            precision=self.precision,
            epsilon=0
        )
        best_solution, best_cost = opti_problem.solve()
        info_dict = opti_problem.make_info_dict(*best_solution)
        return info_dict


    def _find_from_topology_dict_comm_link(self, server_swarm_dict):
        """
        Find all communication links between neighboring swarms.
        Links are directed from lower swarm ID to higher swarm ID.
        """
       # if "comm_link_ids" not in server_swarm_dict[sw_id]:
        #    server_swarm_dict[sw_id]["comm_link_ids"] = []

        server_swarm_keys = sorted(server_swarm_dict.keys())
        for sw_id in server_swarm_keys:
            if sw_id + 1 in server_swarm_dict:
                current_swarm_info = server_swarm_dict[sw_id]
                next_swarm_info = server_swarm_dict[sw_id + 1]
                print(f"Swarm {sw_id} info: {current_swarm_info}")
                print(f"Swarm {sw_id + 1} info: {next_swarm_info}")
                        
                for srv_id in current_swarm_info["server_ids"]:
                    for srv_id_next in next_swarm_info["server_ids"]:
                        for srv in self.servers:
                            if srv.server_id == srv_id:
                                for comm_link in self.communication_links:
                                    if isinstance(comm_link.from_entity, Server) and isinstance(comm_link.to_entity, Server):

                                        if comm_link.from_entity.server_id == srv_id and comm_link.to_entity.server_id == srv_id_next:
                                            self.reduced_communication_links.append(comm_link)

    def _create_full_swarms(self, server_swarm_dict):
        """
        Create swarms with all servers and clients.
        """
        """
        Create a full swarm with all clients and their outgoing communication links.
        """
        all_clients_swarm = Swarm(swarm_id=0, servers=[], clients=self.clients, comm_links=[])
        
        for client in self.clients:
            for link in self.reduced_communication_links:
                if link.from_entity == client:
                    all_clients_swarm.comm_links.append(link)
        
        self.swarms.append(all_clients_swarm)

        for sw_id, swarm_info in server_swarm_dict.items():
            servers = []
            comm_links = []
            for srv_id in swarm_info["server_ids"]:
                for srv in self.servers:
                    if srv.server_id == srv_id:
                        servers.append(srv)
                        
                for link in self.reduced_communication_links:
                    if link.from_entity in servers:
                        comm_links.append(link)
            
            swarm = Swarm(swarm_id=sw_id, servers=servers, clients=[], comm_links=comm_links)
            
            self.swarms.append(swarm)


        for client in self.clients:
            client.add_swarms(self.swarms)


    def _find_comm_link_between_two_servers(self, server_id_tuple):
        """
        Find all communication links between servers.
        """
        links = []
        for link in self.communication_links:
            if isinstance(link.from_entity, Server) and isinstance(link.to_entity, Server):
                if (link.from_entity.server_id, link.to_entity.server_id) == server_id_tuple:
                    links.append(link)
        return links

    
    def _find_client_comm_links_from_topology_dict(self, server_swarm_dict):
        """
        Find all communication links between clients and servers.
        """
        server_swarm_keys = sorted(server_swarm_dict.keys())
        min_swarm_id = min(server_swarm_dict.keys())
        max_swarm_id =  max(server_swarm_dict.keys())

        current_swarm_info = server_swarm_dict[min_swarm_id]
        for srv_id in current_swarm_info["server_ids"]:
            for client in self.clients:
                for srv in self.servers:
                    if srv.server_id == srv_id:
                        for comm_link in self.communication_links:
                            if isinstance(comm_link.from_entity, Client) and isinstance(comm_link.to_entity, Server):
                                if comm_link.from_entity.client_id == client.client_id and comm_link.to_entity.server_id == srv_id:
                                    self.reduced_communication_links.append(comm_link)


        current_swarm_info = server_swarm_dict[max_swarm_id]
        for srv_id in current_swarm_info["server_ids"]:
            for client in self.clients:
                for srv in self.servers:
                    if srv.server_id == srv_id:
                        for comm_link in self.communication_links:
                            if isinstance(comm_link.from_entity, Server) and isinstance(comm_link.to_entity, Client):
                                if comm_link.to_entity.client_id == client.client_id and comm_link.from_entity.server_id == srv_id:
                                    self.reduced_communication_links.append(comm_link)                            

    
    def run_time_steps(self, steps=1000, time_quantum=0.1, start_time=0.1):
        """
        Runs a time-stepped simulation. For each step:
          1) Process time step on each comm link
          2) Process time step on each server
          3) Process time step on each client
        Returns the final time after all steps.
        """
        current_time = start_time
        for _ in range(steps):
            for swarm in reversed(self.swarms):
                swarm.process_time_step(time_quantum=time_quantum, current_time=current_time)
            current_time += time_quantum
        return current_time

    # -------------------------------------------------------------------------
    # MIP to assign servers -> swarms (original example). Now it only runs 
    # if you are not already giving swarms from outside, or if you want
    # to reassign them. 
    # -------------------------------------------------------------------------
    def solve_mip_assign_swarms(self, num_swarms=2):
        """
        Solve a mixed-integer problem to assign servers to `num_swarms` swarms.
        If self.swarms is already set, you can either skip or re-run to reassign.
        """
        model = pulp.LpProblem("Swarm_Assignment", pulp.LpMinimize)

        # Create binary vars x[s, w] = 1 if server s in swarm w
        x = {}
        for s_idx, srv in enumerate(self.servers):
            for w in range(num_swarms):
                x[(s_idx, w)] = pulp.LpVariable(f"x_{s_idx}_{w}", cat=pulp.LpBinary)

        # Constraint: each server belongs to exactly one swarm
        for s_idx, srv in enumerate(self.servers):
            model += pulp.lpSum(x[(s_idx, w)] for w in range(num_swarms)) == 1

        # Build server-server latency map
        server_server_latency = {}
        for link in self.communication_links:
            if isinstance(link.endpoint_a, Server) and isinstance(link.endpoint_b, Server):
                s_a = link.endpoint_a.server_id
                s_b = link.endpoint_b.server_id
                server_server_latency[(s_a, s_b)] = link.latency
                server_server_latency[(s_b, s_a)] = link.latency

        # Objective: sum( x[s_a,w]*x[s_b,w]*latency(s_a, s_b) ), for s_a < s_b
        cost_terms = []
        for s_a_idx, srvA in enumerate(self.servers):
            for s_b_idx, srvB in enumerate(self.servers):
                if s_b_idx <= s_a_idx:
                    continue
                lat = server_server_latency.get((srvA.server_id, srvB.server_id), 0.0)
                for w in range(num_swarms):
                    cost_terms.append(lat * x[(s_a_idx, w)] * x[(s_b_idx, w)])
        model += pulp.lpSum(cost_terms), "Total_Latency_Cost"

        # Solve
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        # Build new swarm structure
        new_swarms = {w: [] for w in range(num_swarms)}
        for s_idx, srv in enumerate(self.servers):
            for w in range(num_swarms):
                if pulp.value(x[(s_idx, w)]) == 1:
                    new_swarms[w].append(srv)

        # Convert to a list of Swarm objects
        self.swarms = []
        for w, srv_list in new_swarms.items():
            # Here, no clients or specific comm_links assigned. You can assign them if desired.
            swarm_obj = Swarm(swarm_id=w, servers=srv_list, clients=[], comm_links=[])
            self.swarms.append(swarm_obj)

        print("[INFO] MIP solved. Swarm assignments:")
        for sw in self.swarms:
            s_ids = [sv.server_id for sv in sw.servers]
            print(f" - Swarm {sw.swarm_id}: Servers {s_ids}")

    def compute_average_latency(self):
        """
        Compute the average latency across all communication links in the system.
        """
        latency = 0
        for swarm in self.swarms:
            latency =+ swarm.calucate_average_latency_servers()
            latency =+ swarm.calculate_avergae_latencies_comm_links()
        return latency

    def run_simulation_iteration(self, iteration_index=0):
        """
        Example: do some single-iteration logic. Here, we compute average latency.
        """
        avg_latency = self.compute_average_latency()
        print(f"[Iteration {iteration_index}] Average latency: {avg_latency:.2f}")
        return avg_latency

    def simulate_swarms_backwards(self, num_steps=2):
        """
        Placeholder for 'simulating the swarms backwards.'
        This is a no-op here, just demonstrating the idea.
        """
        for step in range(num_steps):
            print(f"[Backward Step {step}] Reversing swarm state or time.")
        print("[INFO] Completed backward simulation steps.")

    def create_throughput_balanced_swarms(self, num_swarms, decoder_blocks):
        """
        Create swarms that balance throughput across servers.
        Each server's throughput decreases with the number of assigned decoder blocks.
        :param num_swarms: Number of swarms to create.
        :param decoder_blocks: List of decoder blocks to assign.
        """
        # Initialize swarms
        swarms = {i: [] for i in range(1, num_swarms+1)}
        server_throughputs = {srv: srv.base_throughput for srv in self.servers}
        swarm_decoder_blocks_num = {i: 0 for i in range(1, num_swarms+1)}
        # Sort servers by base throughput in descending order
        sorted_servers = sorted(self.servers, key=lambda srv: srv.base_throughput, reverse=True)

        # Assign servers to swarms
        for server in sorted_servers:
            # Find the swarm with the least total throughput
            best_swarm_id = min(swarms, key=lambda sid: sum(server_throughputs[srv] for srv in swarms[sid]))
            # Assign the server to this swarm
            swarms[best_swarm_id].append(server)

        # Ensure each swarm gets at least one decoder block
        for swarm_id in swarms:
            swarm_decoder_blocks_num[swarm_id] += 1

        # Assign remaining decoder blocks to servers in a round-robin fashion
        remaining_blocks = self.num_decoder_blocks - num_swarms
        for block in range(remaining_blocks):
            # Assign the block to each server in parallel
            total_throughput = {}
            for swarm_id, servers in swarms.items():
                # Calculate the current actual throughput of both swarms
                total_throughput[swarm_id] = sum(server_throughputs[srv] for srv in servers)

            best_swarm_id = max(total_throughput, key=total_throughput.get)
            # Assign the block to the server in this swarm
            swarm_decoder_blocks_num[best_swarm_id] += 1

            # Recalculate server throughputs based on degradation factor
            for server in swarms[best_swarm_id]:
                server_throughputs[server] = server.base_throughput - server.degrading_factor * swarm_decoder_blocks_num[best_swarm_id]


        # Convert to a list of Swarm objects
        self.swarms = []
        # Create a dictionary with swarm_id as key, and entries for number of decoder blocks and server ids
        swarm_info = {}
        for swarm_id, srv_list in swarms.items():
            swarm_info[swarm_id] = {
            "num_decoder_blocks": swarm_decoder_blocks_num[swarm_id],
            "server_ids": [srv.server_id for srv in srv_list]
            }

        print("[INFO] Swarm information:")
        for sw_id, info in swarm_info.items():
            print(f" - Swarm {sw_id}: {info}")

        return swarm_info



