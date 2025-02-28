import random
import numpy as np
import networkx as nx
import os
import signal
import json
from Server import Server
from Communicationlink import CommunicationLink as CL 


class Client:
    def __init__(self, 
                 client_id, 
                 num_stages,
                 max_token_length=None,
                 min_token_length=None,
                 seed=None,
                 job_launch_probability=0.6,
                 external_job_launch=False,
                 routing_policy=None,
                 prompt_mode = None,
                 current_time = 0.1,
                 swarms = None,
                 config_filename='config_simulation.json'):
        """
        :param client_id:               Unique identifier for the client.
        :param server_commLink_Topology: A directed graph (e.g., a NetworkX DiGraph)
                                         that includes cost and capacity attributes 
                                         on edges for min_cost_flow to work.
        :param max_token_length:        Maximum number of tokens for random job generation.
        :param min_token_length:        Minimum number of tokens for random job generation.
        :param seed:                    Optional random seed for reproducibility.
        :param job_launch_probability:  Probability of launching a new job at each time step
        :param external_job_launch:     Flag to indicate if jobs should be launched externally.
        :param routing_policy:          e.g., "min_cost_flow", "throughput", or "random".
        :param config_filename:         Name of the JSON config file. 
        """

        # 1) Load global simulation config (example usage)
        config_path = os.path.join(os.path.dirname(__file__), config_filename)
        with open(config_path, 'r') as config_file:
            config_simulation = json.load(config_file)

        config_client_properties = config_simulation["client_properties"]

        # 2) Store basic parameters
        self.client_id = client_id
        self.id = client_id
        self.topology = None
        self.num_stages = num_stages
        self.job_launch_probability = job_launch_probability
        self.external_job_launch = external_job_launch
        self.routing_policy = routing_policy if routing_policy is not None else config_simulation["routing_policy"]
        self.prompt_mode = prompt_mode if prompt_mode is not None else config_client_properties["prompt_mode"]
        self.d_model = config_simulation["d_model"]
        self.precision = config_simulation["precision"]

        self.launch_in_next_time_step = False

        self.is_dropped_out = False # Future purposes

        # 3) Determine token length bounds
        if (max_token_length is not None) and (min_token_length is not None) and (min_token_length < max_token_length):
            self.max_token_length = max_token_length
            self.min_token_length = min_token_length
        else:
            # Fallback to config defaults
            self.max_token_length = config_client_properties["token_length_range"][1]
            self.min_token_length = config_client_properties["token_length_range"][0]

            self.max_sequence_length = config_simulation["system_properties"]["max_sequence_length"]

        # 4) Random seed control
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.swarms = swarms    

        # 5) Misc. state variables
        
        self.client_node_label = f"client_{self.id}"
        self.sink_node_label   = "SINK_in"   # or whatever node label is your final sink
        self.outgoing_job_iterations = []
        self.job_metadata = {} 
        self.round_robin_dict = {}
        self.incoming_job_iterations = []  
        self.routing_history = []     
        # A global iteration counter to keep iteration IDs unique
        self.global_iteration_counter = 0  # we'll use this in _get_next_iteration_id()


            # Possibly add edges from client to servers (if that is how your scenario is structured)

        self.current_time = 0.1
        self.time_quantum = 0.0001

    @classmethod
    def from_existing_client(cls, existing_client, client_id=None, routing_policy=None):
        """
        Alternative constructor to initialize a new Client based on an existing Client instance.

        :param existing_client: An existing Client instance to copy parameters from.
        :param client_id:       Unique identifier for the new client (optional).
        :param routing_policy:  New routing policy override (optional).
        :return:                A new Client instance.
        """
        return cls(
            client_id=client_id if client_id is not None else existing_client.id,
            server_commLink_Topology=existing_client.topology,
            max_token_length=existing_client.max_token_length,
            min_token_length=existing_client.min_token_length,
            seed=None,  # Optionally, you can choose to copy the seed or not
            job_launch_probability=existing_client.job_launch_probability,
            external_job_launch=existing_client.external_job_launch,
            routing_policy=routing_policy if routing_policy is not None else existing_client.routing_policy,
            config_filename='config_simulation.json'  # Assuming same config file is used
        )
    
    def add_swarms(self, swarms):
        self.swarms = swarms

    def add_topology(self, topology):
        """
        Add a network topology to the client.
        """ # Assign the topology to self.topology

        # 6) Ensure the client node is in the topology
        if self.client_node_label not in topology.nodes:
            raise ValueError(f"Client node {self.client_node_label} is not in the topology.")
        self.topology = topology

    def launch_job_stop(self):
        """
        Stop the client from launching new jobs in the future (by probability).
        """
        self.job_launch_probability = 0.0

    def process_time_step(self,
                          current_time,
                          time_quantum,
                          default_token_buffer_alloc=512,
                          default_KV_cache_alloc=512):
        """
        At each discrete simulation time step, the client:
          1) Potentially launches a new job (unless external_job_launch is True).
          2) Handles re-occurring jobs from the sink (by creating new JobIteration).
          3) Places newly created iterations into 'outgoing_job_iterations'.

        :param current_time:              The current simulation time (or discrete time step).
        :param time_quantum:              (Not used here, but included if needed).
        :param incoming_jobs_from_sink:   List of Job objects returning from the sink node.
        :param outgoing_job_iterations:   A list where we place newly created JobIterations 
                                          for the next server.
        :param default_token_buffer_alloc: 
                                          Default token buffer allocation if not otherwise specified.
        :param default_KV_cache_alloc:
                                          Default KV cache allocation if not otherwise specified.
        """
        print(f"Client {self.client_id}: Processing time step at time {current_time}")
        self.current_time = current_time
        self.time_quantum = time_quantum
        # 1) Possibly launch a new job
        if not self.external_job_launch and self.should_launch_new_job(current_time) and self.launch_in_next_time_step == False:
            print(f"Client {self.client_id}: Launching a new job at time {current_time}")
            new_iter1 = self.start_new_job(current_time,
                                          default_token_buffer_alloc=default_token_buffer_alloc,
                                          default_KV_cache_alloc=default_KV_cache_alloc)

        if self.launch_in_next_time_step:    
            print(f"Client {self.client_id}: REaunching a new job at time {current_time}")

            new_iter = self.start_new_job(current_time,
                                          default_token_buffer_alloc=default_token_buffer_alloc,
                                          default_KV_cache_alloc=default_KV_cache_alloc)
            if new_iter is not None:
                self.launch_in_next_time_step = False

        #2)    push outgoing job_iterations
        self.push_outgoing_jobs()

        # 3) Check the incoming queue for re-occurring jobs (e.g. from the sink).
        #    Typically, you'd remove them from 'incoming_jobs_from_sink', 
        #    create a new iteration, and push that iteration out again.
        while self.incoming_job_iterations:
            next_job_iteration = self.process_throughpass_of_job(self.incoming_job_iterations.pop(0), current_time)  # remove one job
            # create a new iteration
            # push it to the outgoing queue
            if next_job_iteration is not None:
                self.outgoing_job_iterations.append(next_job_iteration)

    def process_throughpass_of_job(self, job_iteration, current_time):
        """
        Process a job iteration that has returned to the client. 
        Update metadata, possibly update the job's status, 
        and create a new iteration if the job is not yet completed.

        :param job_iteration: The incoming JobIteration that has just reached the client.
        :param current_time:  The current simulation time.
        :return: 
            A new JobIteration (if job is still ongoing), 
            or None if the job has completed.
        """
        job = job_iteration.job
        job_id = job.job_id
        self.job_metadata[job_id]["total_tokens"] += job_iteration.token_in_iteration
        
        if job.status == "completed" and self.job_metadata[job_id]["end_time"] is not None:
            return None
        
        if len(self.job_metadata[job_id]["iteration_end_times"]) >= len(self.job_metadata[job_id]["iteration_start_times"])+2:
            print("weird")
        # 1) Mark the 'end' of this iteration in metadata
        self.job_metadata[job_id]["iteration_end_times"].append(current_time)

        # 2) Check if the job transitions from "prompt" to "decoding"
        #    or if it has completely finished ("completed").
        #    This logic assumes you track total tokens processed in self.job_metadata[job_id]["total_tokens"]
        #    and that job.token_prompt_length / job.token_final_length define the boundary.
        if job.status == "initialization":
            print(f"Client {self.client_id}: Job {job_id} is transitioning from initialization to prompt.")
            self.job_metadata[job_id]["start_time"] = current_time
            job.status = "prompt"

        if (self.job_metadata[job_id]["total_tokens"] >= job.token_prompt_length 
            and job.status == "prompt"):
            print(f"Client {self.client_id}: Job {job_id} is transitioning from prompt to decoding.")
            job.status = "decoding"
            job.token_buffer_alloc = 1 


        # If we've now processed (prompt + final) tokens, it’s completed
        if (self.job_metadata[job_id]["total_tokens"] >= job.token_prompt_length + job.decoding_token_length
            and job.status != "completed"):
            print(f"Client {self.client_id}: Job {job_id} has completed.")
            job.status = "completed"
            

        # 3) If the job is completed, do not launch a new iteration
        if job.status == "completed" and self.job_metadata[job_id]["end_time"] is None:
            new_iter_id = self._get_next_iteration_id()
            self.finalize_job(job_id, current_time)
            return JobIteration(job=job, iteration_id=new_iter_id, token_in_iteration=0)
        


        # 4) Otherwise, create a new iteration to continue the job
        new_iter_id = self._get_next_iteration_id()
        if job.status == "prompt" and self.prompt_mode == "all_together":
            # Allocate buffer tokens for the next iteration
            new_iter = JobIteration(job=job, iteration_id=new_iter_id, token_in_iteration=job.token_prompt_length)
        elif job.status == "prompt" and (self.prompt_mode == "individual" or self.prompt_mode == "individual"):
            new_iter = JobIteration(job=job, iteration_id=new_iter_id, token_in_iteration=1)
        elif job.status == "decoding":
            new_iter = JobIteration(job=job, iteration_id=new_iter_id, token_in_iteration=1)    


        # Record the 'start' time of this new iteration in metadata
        self.job_metadata[job_id]["iteration_start_times"].append(current_time)

        return new_iter

    def finalize_job(self, job_id, current_time):
        """
        Finalize a job that has completed.
        """
        if self.job_metadata[job_id]["end_time"] is  None:
            self.job_metadata[job_id]["end_time"] = current_time
        print("========================END=OF=JOB===================================")
        print(f"Client {self.client_id}: Job {job_id} completed at time {current_time} with prompt tokens {self.job_metadata[job_id]['prompt_tokens']} and total tokens {self.job_metadata[job_id]['total_tokens']}")
        print(f"Route: {self.job_metadata[job_id]['route']}")
        print(f"Iteration start times: {self.job_metadata[job_id]['iteration_start_times']}")
        print(f"Iteration end times: {self.job_metadata[job_id]['iteration_end_times']}")
        print(f"Total time: {current_time - self.job_metadata[job_id]['start_time']}")
        print(f"total time for calc: {self.job_metadata[job_id]['end_time'] - self.job_metadata[job_id]['start_time']}")
        print("=====================================================================")

    def start_new_job(self, current_time, default_token_buffer_alloc=512, default_KV_cache_alloc=512):
        """
        Create a brand new Job and place the first JobIteration in self.outgoing_job_iterations.
        Returns the newly created JobIteration (so you can also place it into an external queue).
        """
        new_job_id = f"job_{self.id}_{int(current_time)}_{random.randint(1,9999)}"
        
        new_job = self.launch_job(
            job_id=new_job_id,
            default_token_buffer_alloc=default_token_buffer_alloc,
            default_KV_cache_alloc=default_KV_cache_alloc
        )
        if new_job is None:
            return None 

        first_iter_id = self._get_next_iteration_id()
        first_iter = JobIteration(job=new_job, iteration_id=first_iter_id)
        self.outgoing_job_iterations.append(first_iter)

        # Initialize metadata for the new job
        self.job_metadata[new_job_id] = {
            "start_time": current_time,
            "end_time": None,
            "prompt_tokens": new_job.token_prompt_length,
            "total_tokens": 0,
            "iteration_start_times": [],
            "iteration_end_times": [],
            "route": new_job.route["path"]
        }
        return first_iter


    def should_launch_new_job(self, current_time):
        """
        Decide whether to launch a new job at this time step.
        Simple example: random with fixed probability.
        """
    
        return (random.random() < self.job_launch_probability)

    def launch_job(self, 
                   job_id, 
                   default_token_buffer_alloc=512, 
                   default_KV_cache_alloc=512):
        """
        Create a new random job, compute a route based on the routing policy, 
        and return a Job object with routing_info set.
        """
        # 1) Random prompt/final lengths
        prompt_len, decoding_len = self._create_random_job_lengths()

        
        # 2) Create the Job
        new_job = Job(job_id=job_id,
                      client_id=self.id,
                      token_prompt_length=prompt_len,
                      token_decoding_lengeth=decoding_len,
                      token_buffer_alloc=prompt_len,
                      KV_cache_alloc=self.max_sequence_length)

        max_alloc = max(self.max_sequence_length+1, 3*prompt_len)*self.d_model*self.precision//8
        # 3) Solve for route + cost
        if self.routing_policy == "min_cost_max_flow":
            path, path_obj, cost = self._route_job_via_min_cost_flow(max_alloc)
        elif self.routing_policy == "round_robin":
            path, path_obj, cost = self._route_job_via_interleaved_round_robin()
        elif self.routing_policy == "random":
            # Very naive random path: pick any node or just store a single-step. 
            # (You probably want a real random path that leads to sink, for real usage.)
            path = [self.client_node_label, self.sink_node_label]
            cost = 0
        else:
            # Default fallback or error
            path = [self.client_node_label, self.sink_node_label]
            cost = float('inf')

        if cost == float('inf'):

            print(f"Client {self.client_id}: Job {job_id} has been dropped out.")
            self.launch_in_next_time_step = True
            return None
        # 4) Attach route info
        new_job.add_route({
            "path": path,
            "path_objects": path_obj,
            "cost": cost,
            "current_server_index": 0  # index of which server to visit next
        })
        self.routing_history.append({
            "job_id": job_id,
            "path": path,
            "cost": cost
        })

        return new_job

    def _create_random_job_lengths(self):
        """
        Randomly create (prompt_length, final_length) within configured bounds.
        """
        # Example: force total tokens <= self.max_token_length
        prompt_len = random.randint(self.min_token_length, max(self.min_token_length, self.max_token_length // 2))
        decoding_len  = random.randint(1, max(0, self.max_token_length - prompt_len))
        return (prompt_len, decoding_len)

  
  
  
    def _init_round_robin(self):
        self.round_robin_dict = {f"stage{self.swarms[i].swarm_id}": {} for i in range(len(self.swarms))}

        # Identify servers for each stage dynamically
        for i, swarm in enumerate(self.swarms):
            if i == 0:
                continue  # Skip the first swarm (swarm_0)
            stage_label = f"stage{swarm.swarm_id}"
            self.round_robin_dict[stage_label]["total_weight"] = 0  # Initialize total_weight with 0
            for server in swarm.servers:       
                srv_id = server.server_id
                throughput = int(server.calculate_average_throughput())
                self.round_robin_dict[stage_label][srv_id] = {
                "expected_throughput": throughput,
                "weight": 0,
                "choosen": 0
                }
        self.calculate_round_robin_weights()
        self.sum_round_robin_weights()
  
    # Calculate weights for each server
    def calculate_round_robin_weights(self):
        for stage_label, stage_dict in self.round_robin_dict.items():
            throughputs = [info["expected_throughput"] for key, info in stage_dict.items() if isinstance(info, dict) and key != "total_weight"]
            gcd = np.gcd.reduce(throughputs) if throughputs else 1
            for srv_id, info in stage_dict.items():
                if isinstance(info, dict):
                    info["weight"] = info["expected_throughput"] // gcd

    def sum_round_robin_weights(self):
        """
        Sum up all the weights for each stage and store it in round_robin_dict[stage]["total_weight"].
        """
        for stage_label, stage_dict in self.round_robin_dict.items():
            total_weight = sum(info["weight"] for key, info in stage_dict.items() if key != "total_weight")
            self.round_robin_dict[stage_label]["total_weight"] = total_weight
            
    def _route_job_via_interleaved_round_robin(self):
        """
        Interleaved round-robin routing based on each server's expected throughput.
        Uses self.round_robin_dict, which maps "stageX" -> {server_id -> {...}, "total_weight": ...}.

        Returns: (expanded_path, cost)
        """

        # If round_robin_dict is not yet initialized, do so
        if not self.round_robin_dict:
            self._init_round_robin()

        # --------------------------------------------------------
        # 1) Select servers in interleaved round-robin style
        # --------------------------------------------------------
        selected_servers = []

        for stage_label, stage_dict in self.round_robin_dict.items():
            # Skip stage if total_weight is missing or 0
            if "total_weight" not in stage_dict or stage_dict["total_weight"] == 0:
                continue

            # Find the minimum "choosen" count among all servers (ignore "total_weight")
            min_choosen = min(
                info["choosen"]
                for sid, info in stage_dict.items() 
                if sid != "total_weight"
            )

            # Collect servers that have that min_choosen
            candidates = [
                sid
                for sid, info in stage_dict.items()
                if sid != "total_weight" and info["choosen"] == min_choosen
            ]

            if not candidates:
                # No server available in this stage; skip
                continue

            # Pick the first candidate (could randomize if you prefer)
            selected_server_id = candidates[0]
            selected_servers.append(selected_server_id)

            # Increase its "choosen" count
            stage_dict[selected_server_id]["choosen"] += 1

            # Check if we've reached the stage's total_weight
            total_choosen = sum(
                info["choosen"]
                for sid, info in stage_dict.items()
                if sid != "total_weight"
            )
            if total_choosen >= stage_dict["total_weight"]:
                # Reset all "choosen" in this stage
                for sid, info in stage_dict.items():
                    if sid != "total_weight":
                        info["choosen"] = 0

        # --------------------------------------------------------
        # 2) Construct a path: client -> [servers_in/out] -> sink
        # --------------------------------------------------------
        if not selected_servers:
            # If no servers were selected, fallback path with infinite cost
            fallback_path = [self.client_node_label, self.sink_node_label]
            return self.find_full_route(fallback_path), float('inf')

        path = [self.client_node_label]
        for srv_id in selected_servers:
            path.append(f"server_{srv_id}_in")
            path.append(f"server_{srv_id}_out")
        path.append(self.sink_node_label)

        # --------------------------------------------------------
        # 3) Calculate cost as 1 / sum_of_throughputs (or any desired metric)
        # --------------------------------------------------------
        total_throughput = 0
        for srv_id in selected_servers:
            # Find which stage_label holds this srv_id (we can't assume stage == server_id)
            for st_label, st_dict in self.round_robin_dict.items():
                if srv_id in st_dict and isinstance(st_dict[srv_id], dict):
                    thr = st_dict[srv_id].get("expected_throughput", 0)
                    total_throughput += thr
                    break

        if total_throughput <= 0:
            cost = float('inf')
        else:
            cost = 1.0 / total_throughput

        # Expand path into actual route edges if needed
        full_route, path_objects = self.find_full_route(path)
        return full_route, path_objects, cost

    def find_full_route(self, intermediate_nodes):
        """
        Given a list of intermediate nodes, find a full route from the client node to the sink node.
        
        :param intermediate_nodes: List of intermediate node labels to include in the route.
        :return: A list of node labels representing the full route.
        """
        full_route = [self.client_node_label]
        current_node = self.client_node_label

        for next_node in intermediate_nodes:
            if not nx.has_path(self.topology, current_node, next_node):
                raise ValueError(f"No path found from {current_node} to {next_node}")
            path_segment = nx.shortest_path(self.topology, current_node, next_node)
            full_route.extend(path_segment[1:])  # Skip the first node to avoid duplication
            current_node = next_node

        if not nx.has_path(self.topology, current_node, self.sink_node_label):
            raise ValueError(f"No path found from {current_node} to {self.sink_node_label}")
        path_segment = nx.shortest_path(self.topology, current_node, self.sink_node_label)
        full_route.extend(path_segment[1:])  # Skip the first node to avoid duplication
        
        path_objects = self.path_translate_to_objects(full_route)
        return full_route, path_objects



    
    def _update_weighted_server_list(self):
        """
        Build or rebuild the weighted list of server IDs, repeated
        according to each server's actual throughput.
        """
        weighted_list = []
        
        # For each server, repeat its ID int(throughput) times
        for srv in self.servers:
            weight = int(srv.get_actual_throughput_for_job())
            if weight <= 0:
                continue  # Skip servers with zero or negative throughput
            weighted_list.extend([srv.server_id] * weight)
        
        # Fallback if every server had zero throughput or no servers exist
        if not weighted_list and self.servers:
            weighted_list = [self.servers[0].server_id]  # Fallback to at least one server
        
        self.weighted_server_list = weighted_list
        # Optionally reset round-robin index each time or keep it
        self.wrr_index = 0

    from typing import List

    def update_topology_for_min_cost_max_flow(self, G, memory_max_alloc):
        """
        G:            a networkx.DiGraph
        servers:      list of server objects, each with:
                        - server_id
                        - get_throughput()
                        - cache_memory_capacity
                        - all_memory_allocated
                    (plus any other fields you need)
        comm_links:   list of communication link objects, each with:
                        - link_id
                        - source_id  (ID of the server that sends)
                        - target_id  (ID of the server that receives)
                        - link_throughput (or link_latency, etc.)
                    (plus any other fields you need)
        memory_max_alloc:  A reference capacity for normalizing memory usage.

        This function updates:
        - The internal edges of each server: (server_in -> server_out)
        - The edges representing comm_links: (server_out -> other_server_in)
        with updated 'cost' and 'capacity' attributes, based on throughput
        and memory usage, so min_cost_flow can be re-run with fresh values.
        """
        servers = []
        comm_links = []

        for u, v, data in G.edges(data=True):
            if "object" in data:
                obj = data["object"]
            if isinstance(obj, Server):
                servers.append(obj)
            elif isinstance(obj, CL):
                comm_links.append(obj)
        # 1) Update each server's "internal" edge
        for server in servers:
            # Example naming convention:
            server_in  = f"server_{server.server_id}_in"
            server_out = f"server_{server.server_id}_out"

            # For server processing, you might define cost as an inverse
            # of server throughput, and capacity as how much memory is left.
            actual_throughput = server.get_actual_throughput_for_job()
            cost = 1e9 if actual_throughput == 0 else 1.0 / actual_throughput
            cap = (server.cache_memory_capacity - server.all_memory_allocated) // memory_max_alloc
            cap = max(cap, 0)  # clamp to non-negative
            cap = int(cap)
            if cost >= 1e5:
                cap = 0  # If cost is too high, don't allow flow

            # Update the internal edge attributes (if it exists in G)
            if G.has_edge(server_in, server_out):
                G[server_in][server_out]['cost'] = cost
                G[server_in][server_out]['capacity'] = cap
            else:
                # Optionally, add the edge if it doesn't exist:
                G.add_edge(server_in, server_out, cost=cost, capacity=cap)

        # 2) Update each communication link's edge
        for link in comm_links:
            # The link presumably goes from <server_out> to <other_server_in>.
            # For example, if link.source_id = "serverA", link.target_id = "serverB",
            # then the edge is ( "serverA_out" -> "serverB_in" ).
            if isinstance(link.from_entity, Server):
                source_node = f"server_{link.from_entity.server_id}_out"
            else:
                source_node = f"client_{link.from_entity.client_id}"

            if isinstance(link.to_entity, Server):
                target_node = f"server_{link.to_entity.server_id}_in"
            else:
                target_node = f"SINK_in"

            # Derive cost & capacity from your link’s throughput/latency.
            # For instance:
            #   cost = 1.0 / link.link_throughput
            #   capacity = link.link_throughput
            # or any other function that reflects network constraints.
            cost = link.get_expected_latency_per_token(self.d_model*self.precision)
           # print(f"cost: {cost}, type: {type(cost)}")
            capacity = 1000000000000000000000 #Big number to simulate infinite capacity
            if cost >= 1e5:
                capacity = 0  # If cost is too high, don't allow flow

            # Update or add the edge in the graph
            if G.has_edge(source_node, target_node):
                G[source_node][target_node]['cost'] = cost
                G[source_node][target_node]['capacity'] = capacity
            else:
                G.add_edge(source_node, target_node, cost=cost, capacity=capacity)

        return G        


    def _route_job_via_min_cost_flow(self, max_alloc):
        flow_value = 1
        client_start_node = self.client_node_label
        sink_end_node = self.sink_node_label

        G_copy = self.update_topology_for_min_cost_max_flow(self.topology, max_alloc)

        # Initialize demands
        nx.set_node_attributes(G_copy, 0, "demand")
        G_copy.nodes[client_start_node]["demand"] = -flow_value
        G_copy.nodes[sink_end_node]["demand"] = flow_value

        # def handler(signum, frame):
        #     raise TimeoutError("Min-cost flow computation timed out")

        try:
            # Setup the alarm for 10 seconds
            #signal.signal(signal.SIGALRM) #(..., handler)
            #signal.alarm(1)

            flow_dict = nx.min_cost_flow(
                G_copy,
                demand="demand",
                capacity="capacity",
                weight="cost"
            )

            # Disable alarm if we get here successfully
            signal.alarm(0)

            # Extract path + compute cost
            path, path_obj = self.extract_path_with_objects(flow_dict, client_start_node, sink_end_node)
            total_cost = nx.cost_of_flow(G_copy, flow_dict, weight="cost")

        except TimeoutError:
            print("Min-cost flow computation timed out")
            flow_dict = None
            path = [client_start_node, sink_end_node]
            path_obj = None
            total_cost = float('inf')

        except nx.NetworkXUnfeasible as e:
            print(f"Min-cost flow computation failed: {e}")
            path = [client_start_node, sink_end_node]
            path_obj = None
            total_cost = float('inf')

        return (path, path_obj, total_cost)

    def _extract_path_from_flow(self, flow_dict, source, sink):
        """
        From a flow_dict, extract a single path from `source` to `sink` where flow = 1.
        If multiple paths might carry flow, this only extracts one path that sees >0 flow.
        """
        visited = set()
        stack   = [(source, [source])]  # (current_node, path_so_far)

        while stack:
            node, path_so_far = stack.pop()
            if node == sink:
                return path_so_far
            for neighbor, f_val in flow_dict[node].items():
            # If that edge carried flow and we haven't visited neighbor yet, follow it
                if f_val > 0 and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path_so_far + [neighbor]))
        # If no path found, fallback
        return [source, sink]

    def extract_path_with_objects(self, flow_dict, source, sink):
        """
        From a flow_dict, extract a single path from `source` to `sink` where flow = 1,
        and also return the objects assigned to the graph edges on this path.
        """
        path = self._extract_path_from_flow(flow_dict, source, sink)
        path_objects = self.path_translate_to_objects(path)
        return path, path_objects
    
    def path_translate_to_objects(self, path):
        """
        Given a path (list of node labels), retrieve the 'object' from each
        edge in the path and store it in a dictionary with numeric keys.

        Returns a dict of the form:
        {
        0: {"type": "server" or "comm_link" or "other", "id": <ID>, "obj": <the object>},
        1: {...},
        2: {...},
        ...
        }
        """
        route_objects = {}
        edges = list(zip(path, path[1:]))

        for i, (u, v) in enumerate(edges):
            if not self.topology.has_edge(u, v):
                # Edge does not exist in the graph
                route_objects[i] = {
                    "type": "unknown",
                    "id": f"{u}->{v}",
                    "obj": None
                }
                continue

            edge_data = self.topology[u][v]
            obj = edge_data.get("object", None)

            if obj is None:
                # No object assigned
                route_objects[i] = {
                    "type": "unknown",
                    "id": f"{u}->{v}",
                    "obj": None
                }
                continue

            # Distinguish by known attributes or classes
            if hasattr(obj, "server_id"):
                # It's likely a Server
                route_objects[i] = {
                    "type": "server",
                    "id": obj.server_id,
                    "obj": obj
                }
            elif hasattr(obj, "link_id"):
                # It's likely a CommunicationLink
                route_objects[i] = {
                    "type": "comm_link",
                    "id": obj.link_id,
                    "obj": obj
                }
            else:
                # Unknown or other object type
                route_objects[i] = {
                    "type": "other",
                    "id": f"{u}->{v}",
                    "obj": obj
                }

        route_objects[i+1] = {
            "type": "client",
            "id": self.id,
            "obj": self
        }
        return route_objects
    
    def path_translate_to_objects2(self, path):
        """
        Given a path (a list of node labels, e.g. ["client_0_out", "serverA_in", "serverA_out", "serverB_in", ...]),
        return a list of dicts that detail which server or comm_link corresponds to each edge
        in the path, in order.

        Each entry in the returned list might look like:
        {"type": "server",     "id": <server_id>, "obj": <server_object>}
        {"type": "comm_link",  "id": <link_id>,   "obj": <comm_link_object>}
        {"type": "other",      "id": <node_info>, "obj": None}  # fallback if no match

        This example assumes you can look up servers by matching (server_id + "_in", server_id + "_out")
        and comm_links by matching (from_entity + "_out", to_entity + "_in").
        """

        route_objects = []
        servers = []
        comm_links = []
        for u, v, data in self.topology.edges(data=True):
            if "object" in data:
                obj = data["object"]
            if isinstance(obj, Server):
                servers.append(obj)
            elif isinstance(obj, CL):
                comm_links.append(obj)

        # 1) Build quick-lookup dictionaries for servers and comm_links
        #    so we can map edges (u, v) to the corresponding object.
        #    Example: servers:  server_in->server_out => server
        server_mapping = {}
        for srv in servers:
            s_in = f"server_{srv.server_id}_in"
            s_out = f"server_{srv.server_id}_out"
            server_mapping[(s_in, s_out)] = srv

        #    Example: comm_links: from_entity_out->to_entity_in => link
        comm_link_mapping = {}
        for link in comm_links:
            source_node = f"server_{link.from_entity}_out"
            target_node = f"server_{link.to_entity}_in"
            if target_node == "server_SINK_in":
                target_node = "SINK_in"
            comm_link_mapping[(source_node, target_node)] = link
            # Add the client itself as the sink node if the target_node is "SINK_in"


        # 2) Iterate over each consecutive edge in the path
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            if (u, v) in server_mapping:
                srv_obj = server_mapping[(u, v)]
                route_objects.append({
                    "type": "server",
                    "id": srv_obj.server_id,
                    "obj": srv_obj
                })
            elif (u, v) in comm_link_mapping:
                link_obj = comm_link_mapping[(u, v)]
                route_objects.append({
                    "type": "comm_link",
                    "id": link_obj.link_id,
                    "obj": link_obj
                })

            else:
                # Possibly handle client->server_in edges, server_out->sink edges, 
                # or other special cases. Here we just add a generic entry:
                route_objects.append({
                    "type": "other",
                    "id": f"{u}->{v}",
                    "obj": None
                })
                print(f"Warning: Unhandled edge type from {u} to {v}. Added as 'other'.")
                
                # Handle the case where the target node is the sink
        route_objects.append({
            "type": "client",
            "id": self.id,
            "obj": self
        })
        return route_objects

    def _get_next_iteration_id(self):
        """
        Simple helper to generate a unique iteration ID by incrementing 
        self.global_iteration_counter.
        """
        self.global_iteration_counter += 1
        return self.global_iteration_counter

    def external_launch_job(self, job_id, default_token_buffer_alloc=512, default_KV_cache_alloc=512):
        """
        If jobs can be launched externally, create a new job and return it 
        (so the external code can handle the next steps).
        """
        new_job = self.launch_job(
            job_id=job_id,
            default_token_buffer_alloc=default_token_buffer_alloc,
            default_KV_cache_alloc=default_KV_cache_alloc
        )
        return new_job

    def add_jobIteration_to_queue(self, job_iteration, outgoing_job_iterations):
        """
        Example method to push an incoming job iteration into the given queue.
        """
        self.incoming_job_iterations.append(job_iteration)

    def start_job(self, job_iteration):
        """
        Redirect this call to add_jobIteration_to_queue, so that any 
        external call to start_job() behaves the same.
        """
        self.add_jobIteration_to_queue(job_iteration, self.outgoing_job_iterations)    

    def push_outgoing_jobs(self):
        """
        Push jobs from outgoing queues to the next server or communication link.
        Only push if the next server and communication link are available (not dropped out).
        """
        intermediate_job_iterations_failed_pushes = []
        while self.outgoing_job_iterations:
            job_iteration = self.outgoing_job_iterations.pop(0)
            job_id = job_iteration.job.job_id
            try:
                next_server = job_iteration.job.route["path_objects"][1]["obj"] # Assume job contains routing info
                comm_link = job_iteration.job.route["path_objects"][0]["obj"]  # Assume job contains comm link info
            except AttributeError as e:
                raise SystemExit(f"Error processing job {job_id}: {e}")
                
    #        assert isinstance(next_server, Server) or next_server is None, f"Expected Server object, got {type(next_server)}"
  #          assert isinstance(comm_link, CL) or comm_link is None, f"Expected CL object, got {type(comm_link)}"
 #           assert next_server.is_dropped_out is False, f"Server {next_server.server_id} is dropped out"
#            assert comm_link.link_state == "UP", f"Comm link {comm_link.link_id} is down"
            # Check if next server and comm link are available
            if next_server and next_server.is_dropped_out is False and comm_link and comm_link.link_state == "UP":
                comm_link.add_job_iteration(job_iteration, self.current_time)  # Forward the job
            else:
            # If not available, requeue the job
                intermediate_job_iterations_failed_pushes.append(job_iteration)
                
        self.outgoing_job_iterations.extend(intermediate_job_iterations_failed_pushes)




    

class Job:
    def __init__(self, job_id, client_id, token_prompt_length, token_decoding_lengeth, token_buffer_alloc, KV_cache_alloc, route=None):
        """
        Initialize a job with the given properties.
        
        :param job_id: Unique identifier for the job.
        :param client_id: ID of the client launching the job.
        :param token_prompt_length: Length of the token prompt for the job.
        :param token_buffer_alloc: Buffer allocation for the job.
        :param KV_cache_alloc: KV cache allocation for the job.
        """
        self.job_id = job_id
        self.client_id = client_id
        self.token_prompt_length = token_prompt_length
        self.decoding_token_length = token_decoding_lengeth # Completion - prompt

        self.route = route  #[{object1_id: object1},{object2_id: object2}]
        
        # Counters and status
        self.all_tokens_sum = 0  
        self.status = "initialization"  # Initial status
        
        # Resource usage or allocations
        self.token_buffer_alloc = token_buffer_alloc 
        self.KV_cache_alloc = KV_cache_alloc

        self.random_test = "random"

        self.current_index = 0

    def change_status(self, new_status):
        """
        Change the status of the job.
        
        :param new_status: New status to assign to the job.
        """
        self.status = new_status    
    
    def add_route(self, routing_info):
        """
        Attach a route to the job.
        
        :param route: The route to attach to the job.
                new_job.add_route({
            "path": path,
            "path_objects": path_obj,
            "cost": cost,
            "current_server_index": 0  # index of which server to visit next
        })
        path_obj = {
                    "type": "server",
                    "id": obj.server_id,
                    "obj": obj
                }
        """
        self.route = routing_info

    def change_random_test(self, new_random_test):
        """
        Change the status of the job.
        
        :param new_status: New status to assign to the job.
        """
        self.random_test = new_random_test    

    def get_next_commLink(self, current_server_id):    
        """
        Get the next communication link in the route.
        
        :param server_id: The server ID that is currently processing the job.
        :return: The next communication link in the route.
        """
        if self.route["path_objects"][self.current_index]["id"]==current_server_id:
            return self.route["path_objects"][self.current_index+1]["obj"]
        else:
            raise ValueError(f"Current server ID {current_server_id} does not match the expected server ID in the route.")
    
    def get_next_server(self, current_server_id):
        """
        Get the next server in the route for the job.
        
        :param current_server_id: The server ID that is currently processing the job.
        :return: The next server ID in the route, or None if the current server is the last one.
        """

        if self.route["path_objects"][self.current_index]["id"]==current_server_id:
            return self.route["path_objects"][self.current_index+2]["obj"]
        else:
            raise ValueError(f"Current server ID {current_server_id} does not match the expected server ID in the route.")
        


    def __repr__(self):
        """
        String representation of the job for debugging and logging.
        """
        return (
            f"Job("
            f"id={self.job_id}, "
            f"client_id={self.client_id}, "
            f"token_prompt_length={self.token_prompt_length}, "
            f"status={self.status}, "
            f"current_iteration_num_token={self.current_iteration_num_token}, "
            f"token_buffer_alloc={self.token_buffer_alloc}, "
            f"KV_cache_alloc={self.KV_cache_alloc})"
            f"random_test={self.random_test}"
        )


class JobIteration:
    """
    Represents a single iteration that operates on a shared Job object.
    Multiple JobIteration instances can reference the same Job.
    """
    def __init__(self, job, iteration_id, token_in_iteration=0):
        """
        :param job: Reference to the shared Job object.
        :param iteration_id: Unique identifier for this iteration.
        """
        self.job = job
        self.iteration_id = iteration_id
        
        # Track iteration-specific usage
        self.token_in_iteration = token_in_iteration
        self.iter_type = "initialization"
        self.synchron_iter_typ_with_job_status()
        self.job.all_tokens_sum += token_in_iteration
        self.current_route_index = 0
        self.reset_index()

    def  synchron_iter_typ_with_job_status(self):
        """
        Synchronize the iteration type with the job status.
        """
        self.iter_type = self.job.status   

    def reset_index(self):
        """
        Reset the current route index to 0.
        """
        self.current_route_index = 0
        self.job.current_index = 0

    def change_jobs_random_test(self, new_random_test):
        """
        Change the status of the job.
        
        :param new_status: New status to assign to the job.
        """
        self.job.change_random_test(new_random_test)

    def check_route_index(self, server_id):
        """
        Check the current route index for the given server ID.
        
        :param server_id: The server ID to check.
        :return: True if the server ID matches the current route index, False otherwise.
        """
        success = False
        for index, server_dict in self.job.route["path_objects"].items():
            if server_dict["type"] == "comm_link":
                continue
            if server_dict["id"] == server_id:
                self.current_route_index = index
                self.job.current_index = index
                success = True
                break
            
        if not success:
            raise ValueError(f"Server ID {server_id} not found in the route.")
            
    def __repr__(self):
        """
        String representation of the job iteration for debugging and logging.
        """
        return (
            f"JobIteration("
            f"iteration_id={self.iteration_id}, "
            f"job_id={self.job.job_id}, "
            f"current_job_status={self.job.status}, "
#            f"token_buffer_use={self.token_buffer_use}, " #Job property -  revision
#            f"KV_cache_growth={self.KV_cache_growth})" #Job property - revision
        )
    

if __name__ == "__main__":
    print("Client, Job, and JobIteration classes defined.")
    # Example usage:
    # 1. Create a Job object
    my_job = Job(job_id=1, client_id=123, token_prompt_length=100, 
                    token_buffer_alloc=2048, KV_cache_alloc=1024)
    
    # 2. Create multiple JobIteration objects that share the same Job
    iteration1 = JobIteration(my_job, iteration_id=1)
    iteration2 = JobIteration(my_job, iteration_id=2)
    
    # 3. Process tokens in iteration1
    iteration1.process_iteration(150)  # halfway through prompt phase
    print(my_job)
    print(iteration1)
    
    # 4. Process tokens in iteration2s
    iteration2.process_iteration(50)  # finishes prompt phase, switches to completion
    print(my_job)
    print(iteration2)
    
    # 5. Process more tokens in iteration1 (now in completion phase)
    iteration1.process_iteration(100)  # might finish the job
    print(my_job)
    print(iteration1)