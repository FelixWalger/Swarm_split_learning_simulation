import math
import random
import numpy as np
import os
import json

class Server:
    """
    Server class designed to operate at every simulation time quantum.
    - Supports adaptive weighting for incoming job queues.
    - Maintains per-job incoming and outgoing queues.
    - Processes jobs with memory management (KV cache, token buffer, model parameters).
    - Tracks time-based metadata for jobs (arrival, processing start, and completion times).
    - Supports throughput updates based on Gaussian distribution and dropout logic.
    - Pushes completed jobs to the next server or communication link when available.
    - Synchronizes with global simulation time.
    """
    def __init__(self,
                 server_id,
                 memory_capacity=None,
                 throughput_dist_std_factor=None,
                 base_throughput=None,
                 degrading_factor=None,
                 decoder_blocks=None,
                 selection_strategy=None,
                 seed=None,
                 config_filename='config_simulation.json'):
        """
        Initialize the server.
        (Parameter documentation omitted for brevity.)
        """
        self.server_id = server_id
        config_path = os.path.join(os.path.dirname(__file__), config_filename)
        with open(config_path, 'r') as config_file:
            config_simulation = json.load(config_file)
        server_properties = config_simulation["server_properties"]
        self.precision = config_simulation["precision"]
        self.d_model = config_simulation["d_model"]

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
        self.throughput_dist_std_factor = throughput_dist_std_factor if throughput_dist_std_factor is not None else 0.1
        self.base_throughput = base_throughput if base_throughput is not None else random.uniform(*server_properties["throughput_range"])
        self.degrading_factor = degrading_factor if degrading_factor is not None else random.uniform(*server_properties["degradation_rate"])
        self.selection_strategy = selection_strategy if selection_strategy is not None else server_properties["selection_strategy"]

        self.cache_memory_capacity = int(memory_capacity) if memory_capacity is not None else int(random.uniform(*server_properties["memory_range_in_GB"]) * (1024 ** 3))  # Convert GB to bytes
        # Initialize decoder blocks.
        # (Assuming each decoder block has an attribute "block_id" and a method get_model_memory_usage().)
        self.decoder_blocks = {} 
        self.num_blocks = 0
        self.lowest_decoder_block_id = None

        # Memory tracking initialization.
        self.kv_mem_used = 0
        self.token_buffer_used = 0
        self.model_param_mem_used = 0
        self.kv_mem_allocated = 0
        self.token_buffer_allocated = 0
        self.model_param_mem_allocated = 0
        self.all_memory_used = 0
        self.all_memory_allocated = 0

        # ---------------------------
        # Dropout / Outage Model Setup
        # ---------------------------
        # Instead of using dropout_prob and dropout_length_dist_params,
        # we use a continuous-time Markov chain with exponential holding times.
        #
        # Sample a mean uptime from the configuration range "down_server_time_range_min".
        # (Interpretation: while the server is UP, it remains UP for an average of T_up seconds.)
        downtime = random.uniform(*server_properties["down_server_time_range_min"])
        self.recovery_rate = 1.0 / downtime  # Rate at which the server fails (goes DOWN).

        # Similarly, sample a mean downtime from "up_server_time_range_min".
        # (Interpretation: while the server is DOWN, it remains DOWN for an average of T_down seconds.)
        uptime = random.uniform(*server_properties["up_server_time_range_min"])
        self.failure_rate = 1.0 / uptime  # Rate at which the server recovers (goes UP).

        # Initialize the state: assume the server starts in the UP state.
        self.is_dropped_out = False
        # Set a very small initial GLOBAL_TIME (or you can set it to 0).
        self.GLOBAL_TIME = 0.001
        self.GLOBAL_TIME_QUANTUM = 0.001

        # Sample the next failure time using an exponential distribution.
        self.next_failure_time = self.GLOBAL_TIME + np.random.exponential(1.0 / self.recovery_rate)
        self.next_recovery_time = None

        # ---------------------------
        # Job Queues and other initialization.
        # ---------------------------
        self.incoming_queues = {}  # {job_id[job_obj]:job_obj,  weight)}
        self.outgoing_queues = {}  # {job_id: [job_obj]}
        self.job_metadata = {}     # Metadata for each job (time tracking, weights, etc.)
        self.current_open_jobs = []
        self.completed_jobs_log = []
        self.memory_usage_tracking = {}

        if decoder_blocks:
            self.assign_list_of_decoder_blocks(decoder_blocks) 

        self.actual_throughput = self.sample_actual_throughput()    

    @classmethod
    def from_existing_server(cls, existing_server, config_filename=None):
        """
        Alternative constructor to initialize a server based on an existing server instance,
        with an option to change the configuration file.
        """
        new_server = cls(
            server_id=existing_server.server_id,
            memory_capacity=existing_server.cache_memory_capacity,
            throughput_dist_std_factor=existing_server.throughput_dist_std_factor,
            base_throughput=existing_server.base_throughput,
            degrading_factor=existing_server.degrading_factor,
            decoder_blocks=[block.copy() for block in existing_server.decoder_blocks.values()],
            config_filename=config_filename if config_filename else existing_server.config_filename
        )
        new_server.kv_mem_used = existing_server.kv_mem_used
        new_server.token_buffer_used = existing_server.token_buffer_used
        new_server.model_param_mem_used = existing_server.model_param_mem_used
        new_server.kv_mem_allocated = existing_server.kv_mem_allocated
        new_server.token_buffer_allocated = existing_server.token_buffer_allocated
        new_server.model_param_mem_allocated = existing_server.model_param_mem_allocated
        new_server.all_memory_used = existing_server.all_memory_used
        new_server.all_memory_allocated = existing_server.all_memory_allocated

        new_server.recovery_rate = existing_server.recovery_rate
        new_server.failure_rate = existing_server.failure_rate
        new_server.is_dropped_out = existing_server.is_dropped_out
        new_server.GLOBAL_TIME = existing_server.GLOBAL_TIME
        new_server.GLOBAL_TIME_QUANTUM = existing_server.GLOBAL_TIME_QUANTUM
        new_server.next_failure_time = existing_server.next_failure_time
        new_server.next_recovery_time = existing_server.next_recovery_time

        new_server.incoming_queues = existing_server.incoming_queues.copy()
        new_server.outgoing_queues = existing_server.outgoing_queues.copy()
        new_server.job_metadata = existing_server.job_metadata.copy()
        new_server.current_open_jobs = existing_server.current_open_jobs.copy()
        new_server.completed_jobs_log = existing_server.completed_jobs_log.copy()
        new_server.memory_usage_tracking = existing_server.memory_usage_tracking.copy()

        return new_server
    # -------------------------------------------------------------------------
    #                   DECODER BLOCKS 
    # -------------------------------------------------------------------------
    def assign_list_of_decoder_blocks(self, decoder_blocks):
        """
        Assign a list of decoder blocks to the server.
        """
        for decoder_block in decoder_blocks:
            self.assign_decoder_block(decoder_block)

    def assign_decoder_block(self, decoder_block, allocation = True):
        """
        Assign a decoder block to the server.
        The memory usage for d_model and KV_cache is updated accordingly.
        """
        self.decoder_blocks[decoder_block.block_id] = decoder_block
        self.num_blocks = len(self.decoder_blocks)
        # Example memory usage update. Adjust as needed:
        if allocation:
            print("Here")
            print(self.allocate_model_memory(decoder_block.get_model_memory_usage()))
        print(self.use_model_memory(decoder_block.get_model_memory_usage()))    
        self.lowest_decoder_block_id = min(self.decoder_blocks.keys()) if self.decoder_blocks else None

    def remove_decoder_block(self, decoder_block, release = True):
        """
        Remove a decoder block from the server.
        The memory usage for d_model is updated accordingly.
        """
        del self.decoder_blocks[decoder_block.block_id]
        self.num_blocks = len(self.decoder_blocks)
        # Example memory usage update. Adjust as needed:
        if release:
            self.release_model_memory(decoder_block.get_model_memory_usage())
        self.free_model_memory(decoder_block.get_model_memory_usage())
        self.lowest_decoder_block_id = min(self.decoder_blocks.keys()) if self.decoder_blocks else None    

    # -------------------------------------------------------------------------
    #                   memory management
    # -------------------------------------------------------------------------    

    def allocate_model_memory(self, model_block_num_params, precision = None):
        """
        Reserve memory for the job's model parameters.
        """
        print("model_block_num_params", model_block_num_params)
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        model_mem_alloc = model_block_num_params * precision_bytes
        self.model_param_mem_allocated += model_mem_alloc
        self.all_memory_allocated += model_mem_alloc
        return model_mem_alloc


    def allocate_kv_memory(self, num_tokens_kv_cache_request, precision = None, job_id = None):
        """
        Reserve memory for the job's KV cache.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        kv_mem_alloc = 2 * num_tokens_kv_cache_request *precision_bytes*self.d_model
        self.kv_mem_allocated += kv_mem_alloc
        self.all_memory_allocated += kv_mem_alloc
        if job_id:
            self.job_metadata[job_id]["kv_mem_allocated"] += kv_mem_alloc
        return kv_mem_alloc


    def allocate_token_buffer_memory(self, num_token_buffer_request, precision = None, job_id = None):
        """
        Reserve memory for the job's token buffer.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        tok_buf_mem_alloc = num_token_buffer_request* precision_bytes*self.d_model
        self.token_buffer_allocated += tok_buf_mem_alloc
        self.all_memory_allocated += tok_buf_mem_alloc
        if job_id:
            self.job_metadata[job_id]["token_buffer_allocated"] += tok_buf_mem_alloc
        return tok_buf_mem_alloc

    def use_kv_memory(self, num_tokens_kv_cache_use, precision = None, job_id = None):
        """
        Use actual memory for the job's KV cache.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        kv_mem = 2 * num_tokens_kv_cache_use *precision_bytes*self.d_model
        self.kv_mem_used += kv_mem
        self.all_memory_used += kv_mem
        if job_id:
            self.job_metadata[job_id]["kv_mem_used"] += kv_mem
        return kv_mem


    def use_token_buffer_memory(self, num_token_buffer_use, precision = None, job_id = None):
        """
        Use actual memory for the job's token buffer.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        token_buf_mem = num_token_buffer_use *precision_bytes*self.d_model
        self.token_buffer_used += token_buf_mem
        self.all_memory_used += token_buf_mem
        if job_id:
            self.job_metadata[job_id]["token_buffer_used"] += token_buf_mem
        return token_buf_mem

    def use_model_memory(self, model_block_num_params, precision = None):
        """
        Use actual memory for the job's model parameters.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        model_mem = model_block_num_params *precision_bytes
        self.model_param_mem_used += model_mem        
        self.all_memory_used += model_mem
        return model_mem

    def free_kv_memory(self, release_tokens, precision = None, job_id = None):
        """
        Free memory for the job's KV cache.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        kv_mem = 2 * release_tokens *precision_bytes*self.d_model
        self.kv_mem_used -= kv_mem
        if job_id:
            self.job_metadata[job_id]["kv_mem_used"] -= kv_mem
        self.all_memory_used -= kv_mem
        return kv_mem    


    def free_token_buffer_memory(self, release_tokens, precision = None, job_id = None):
        """
        Free memory for the job's token buffer.
        free is opposite of use.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        token_buf_mem = release_tokens *precision_bytes*self.d_model
        self.token_buffer_used -= token_buf_mem 
        if job_id:
            self.job_metadata[job_id]["token_buffer_used"] -= token_buf_mem
        self.all_memory_used -= token_buf_mem
        return token_buf_mem

    def free_model_memory(self, model_block_num_params, precision = None):
        """
        Free memory for the job's model parameters.
        """
        if precision is None:
            precision = self.precision
        precision_bytes = precision // 8
        model_mem = model_block_num_params *precision_bytes
        self.model_param_mem_used -= model_mem
        self.all_memory_used -= model_mem
        return model_mem

    def release_kv_memory(self, release_tokens, precision = None, job_id = None):
        """
        Release memory for the job's KV cache.
        """
        if precision is None:
            precision = self.precision
        kv_mem = 2 * release_tokens * precision// 8
        self.kv_mem_allocated -= kv_mem
        if job_id:
            self.job_metadata[job_id]["kv_mem_allocated"] -= kv_mem
        self.all_memory_allocated -= kv_mem
        return kv_mem


    def release_token_buffer_memory(self, release_tokens, precision = None, job_id = None):
        """
        Release memory for the job's token buffer.
        """
        if precision is None:
            precision = self.precision
        token_buf_mem = release_tokens * precision// 8
        self.token_buffer_allocated -= token_buf_mem
        self.all_memory_allocated -= token_buf_mem
        if job_id:
            self.job_metadata[job_id]["token_buffer_allocated"] -= token_buf_mem
        return token_buf_mem

    def release_model_memory(self, model_block_num_params, precision = None):     
        """
        Release memory for the job's model parameters.
        """
        if precision is None:
            precision = self.precision
        model_mem = model_block_num_params * precision// 8
        self.model_param_mem_allocated -= model_mem       
        self.all_memory_allocated -= model_mem
        return model_mem
    
    def memory_use_alloc_sync(self):
        """
        Synchronize memeory usage and allocation by ensuring that the allocated memory is not less than the used memory.
        """
        self.kv_mem_allocated = max(self.kv_mem_allocated, self.kv_mem_used)
        self.token_buffer_allocated = max(self.token_buffer_allocated, self.token_buffer_used)
        self.model_param_mem_allocated = max(self.model_param_mem_allocated, self.model_param_mem_used)
        self.all_memory_allocated = max(self.all_memory_allocated, self.all_memory_used)

    def allocate_job_memory(self, job):
        """
        Allocate memory for the job's KV cache and token buffer.
        """
        kv_mem = self.allocate_kv_memory(job.KV_cache_alloc, job_id=job.job_id)
        token_buf_mem = self.allocate_token_buffer_memory(job.token_buffer_alloc, job_id=job.job_id)
        return kv_mem, token_buf_mem
    
    def change_job_memory_allocation(self, job):
        """
        Change memory allocation for the job's KV cache and token buffer.
        """
        kv_mem = self.release_kv_memory(self.job_metadata[job.job_id]["kv_mem_allocated"], job_id=job.job_id)
        kv_mem = self.allocate_kv_memory(job.KV_cache_alloc, job_id=job.job_id)
        token_buf_mem = self.release_token_buffer_memory(self.job_metadata[job.job_id]["token_buffer_allocated"], job_id=job.job_id)
        token_buf_mem = self.allocate_token_buffer_memory(job.token_buffer_alloc, job_id=job.job_id)
        return kv_mem, token_buf_mem

    def free_job_memory(self, job_id):
        """
        Free memory for the job's KV cache and token buffer.
        """
        kv_mem = self.free_kv_memory(self.job_metadata[job_id]["kv_mem_used"], job_id=job_id)
        token_buf_mem = self.free_token_buffer_memory(self.job_metadata[job_id]["token_buffer_used"], job_id=job_id)
        return kv_mem, token_buf_mem

    def release_job_memory(self, job_id):
        """
        Release memory for the job's KV cache and token buffer.
        """
        kv_mem = self.release_kv_memory(self.job_metadata[job_id]["kv_mem_allocated"], job_id=job_id)
        token_buf_mem = self.release_token_buffer_memory(self.job_metadata[job_id]["token_buffer_allocated"], job_id=job_id)
        return kv_mem, token_buf_mem, 

      
        

    # -------------------------------------------------------------------------
    #                        Throughput 
    # -------------------------------------------------------------------------
    def calculate_average_throughput(self, num_blocks=None):
        """
        Calculate average throughput based on assigned decoder blocks.
        avg_throughput = base_throughput / (1 + degrading_factor * num_blocks)
        """
        penalty_factor = 1
        if self.all_memory_used > self.cache_memory_capacity:
            penalty_factor = 10
        if self.num_blocks == 1 or num_blocks == 1:
            return self.base_throughput*penalty_factor
        if num_blocks is None:
            return (self.base_throughput - (self.degrading_factor * self.num_blocks))*penalty_factor
        elif num_blocks > 1:
            return (self.base_throughput - (self.degrading_factor * num_blocks)) * penalty_factor

    def sample_actual_throughput(self):
        """
        Sample the actual throughput using a normal distribution centered on avg_throughput,
        clamped to ±50% of avg_throughput.
        """
        avg_thr = self.calculate_average_throughput()
        std = avg_thr * self.throughput_dist_std_factor
        thr = np.random.normal(loc=avg_thr, scale=std)
        thr = max(avg_thr / 2, min(avg_thr + avg_thr / 2, thr))

        if self.all_memory_used > self.cache_memory_capacity:
            thr = thr*10 # if memory is full, reduce throughput; punshement for overusing memory

        thr = self.GLOBAL_TIME_QUANTUM * math.floor(thr / self.GLOBAL_TIME_QUANTUM)
        return thr
    
    def get_actual_throughput_for_job(self):
        return self.actual_throughput

    # -------------------------------------------------------------------------
    # Updated Dropout / Outage Model Method
    # -------------------------------------------------------------------------
    def check_for_dropout(self):
        """
        Update the server's dropout (outage) state using a continuous-time Markov chain model.
        
        - When the server is UP, it remains UP until GLOBAL_TIME reaches next_failure_time.
          At that moment, the server goes DOWN and a recovery time is sampled.
        - When the server is DOWN, it remains DOWN until GLOBAL_TIME reaches next_recovery_time.
          Then it goes UP and a new failure time is sampled.
        """
        if not self.is_dropped_out:
            # Server is UP. Check if it's time to fail.
            if self.GLOBAL_TIME >= self.next_failure_time:
                self.is_dropped_out = True
                # Sample recovery time: time until recovery ~ Exponential(1/recovery_rate)
                self.next_recovery_time = self.GLOBAL_TIME + np.random.exponential(1.0 / self.recovery_rate)
                print(f"Server {self.server_id} FAILED at time {self.GLOBAL_TIME:.3f}; will recover at {self.next_recovery_time:.3f}")
        else:
            # Server is DOWN. Check if it's time to recover.
            if self.GLOBAL_TIME >= self.next_recovery_time:
                self.is_dropped_out = False
                # Sample the next failure time: time until next failure ~ Exponential(1/recovery_rate)
                self.next_failure_time = self.GLOBAL_TIME + np.random.exponential(1.0 / self.failure_rate)
                print(f"Server {self.server_id} RECOVERED at time {self.GLOBAL_TIME:.3f}; next failure at {self.next_failure_time:.3f}")

    def get_throughput_info(self):
        """
        Return throughput information (actual, max, average).
        """
        avg_thr = self.calculate_average_throughput()
        return {
            "actual": self.actual_throughput,
            "max": self.base_throughput,
            "average": avg_thr
        }

    def check_start_of_server_slot(self):
        """
        check whther the current global slot is also start of a server time slot
        """
        N = int((1/self.actual_throughput )/ self.GLOBAL_TIME_QUANTUM)
        if N*self.GLOBAL_TIME_QUANTUM == self.actual_throughput:
            return True
        else:
            return False
    # -------------------------------------------------------------------------
    #                        Job Queue Management
    # -------------------------------------------------------------------------
    def add_jobIteration_to_queue(self, job_iteration, current_time):
        """
        Add a new job to the incoming queue with an initial weight.
        """
        job = job_iteration.job
        job_iteration.check_route_index(self.server_id)
        self.change_job_memory_allocation(job)
        self.incoming_queues[job.job_id]["in_queue"].append(job_iteration)  
        self.use_token_buffer_memory(job_iteration.token_in_iteration, job_id=job.job_id)
        self.job_metadata[job.job_id]["tokens_processed"] = 0
        self.job_metadata[job.job_id]["number_of_tokens"] = job_iteration.token_in_iteration
        self.job_metadata[job.job_id]["arrival_time"] = current_time
        self.job_metadata[job.job_id]["sum_tokens_over_all_time"] += job_iteration.token_in_iteration
        for i, decoder_block in enumerate(self.decoder_blocks.values()):
            print(f"decoder{i}_take: {decoder_block}")
            self.job_metadata[job.job_id][f"decoder{decoder_block.block_id}_take"] = 0
            self.job_metadata[job.job_id][f"decoder{decoder_block.block_id}_num_of_tok_completed"] = 0
        self.update_queue_weight(job.job_id)
        if job.status == "completed":
            self.terminate_job(job.job_id)

    def create_queues_and_trackinDict_for_job(self, job):
        self.incoming_queues[job.job_id] = {"in_queue": [], "weight": 0}
        self.outgoing_queues[job.job_id] = []
        self.job_metadata[job.job_id] = {
            "arrival_time": self.GLOBAL_TIME,  # Assume GLOBAL_TIME is provided by the System.
            "processing_time": 0,
            "job_end": False,
            "start_time": None,
            "completion_time": 0,
            "number_of_tokens": 0,
            "tokens_processed": 0,
            "last_decoder_block_completion_time": 0,
            "kv_mem_allocated": 0,
            "token_buffer_allocated": 0,
            "kv_mem_used": 0,
            "token_buffer_used": 0,
            "sum_tokens_over_all_time": 0,
            }
        for i, decoder_block in enumerate(self.decoder_blocks.values()):
            print(f"decoder{i}_take: {decoder_block}")
            self.job_metadata[job.job_id][f"decoder{decoder_block.block_id}_take"] = 0
            self.job_metadata[job.job_id][f"decoder{decoder_block.block_id}_num_of_tok_completed"] = 0
        self.update_queue_weight(job.job_id)

    def remove_job_from_queues(self, job_id):
        """
        Remove all references to a job from queues and metadata.
        """
        if job_id in self.incoming_queues:
            del self.incoming_queues[job_id]
        if job_id in self.outgoing_queues:
            del self.outgoing_queues[job_id]
        if job_id in self.job_metadata:
            del self.job_metadata[job_id]
        self.current_open_jobs = [job for job in self.current_open_jobs if job.job_id != job_id]


    def update_queue_weights(self):
        """
        Update the weights of all incoming job queues based on the throughput and job metadata.
        """
    
        for job_id, queue_info in self.incoming_queues.items():
            if queue_info["in_queue"]:
                self.update_queue_weight(job_id)
        

    def update_queue_weight(self, job_id):
        """
        Update the weight of the incoming queue for the specified job.
        :param job_id: The ID of the job whose queue weight is being updated.
        :param new_weight: The new weight to assign to the queue.
        """
        new_weight = self.calculate_job_weight(job_id)
        self.incoming_queues[job_id]["weight"] = new_weight

    def calculate_job_weight(self, job_id):
        """
        Retrieve the job with the highest weight among all queues based on the selection strategy.
        """
        new_weight = 0
        if self.selection_strategy == "FIFO":
            new_weight =  self.calculate_weight_fifo(job_id)
        elif self.selection_strategy == "longest_time_not_seen":
            new_weight = self.calculate_weight_longest_time_not_seen(job_id)
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        #new_normalized_weight = 1-(new_weight / int(self.GLOBAL_TIME / self.GLOBAL_TIME_QUANTUM))
        return new_weight
        
    def calculate_weight_fifo(self, job_id):
        """
        Calculate the weight for a job based on the FIFO strategy.
        """
        time_since_arrival = self.GLOBAL_TIME - self.job_metadata[job_id]["arrival_time"]
        return int(time_since_arrival/self.GLOBAL_TIME_QUANTUM)
             
    def calculate_weight_longest_time_not_seen(self, job_id):
        """
        Calculate the weight for a job based on the longest time not seen.
        """
        time_since_last_seen = self.GLOBAL_TIME - self.job_metadata[job_id]["completion_time"]
        return int(time_since_last_seen/self.GLOBAL_TIME_QUANTUM)

    def get_highest_weight_job(self):
        """
        Retrieve the job with the highest weight among all queues.
        """
        selected_job_id = None
        highest_weight = float("-inf")

        for job_id, queue_info in self.incoming_queues.items():
            queue = queue_info["in_queue"]
            weight = queue_info["weight"]
            if queue and weight > highest_weight:
                highest_weight = weight
                selected_job_id = job_id

        if selected_job_id is not None:
            return self.incoming_queues[selected_job_id]["in_queue"].pop(0)  # Return the highest-weight job
        return None

    # -------------------------------------------------------------------------
    #                        Job Processing
    # -------------------------------------------------------------------------
    def process_time_step_at_decoder(self):
        """
        Process a given number of tokens for a job.
        """
        for decoder_block_id in sorted(self.decoder_blocks.keys()):
            decoder_block = self.decoder_blocks[decoder_block_id]
            if decoder_block.is_processing:
                job_id, done, num_tokens = decoder_block.process_token(self.actual_throughput, quantum=self.GLOBAL_TIME_QUANTUM, global_time=self.GLOBAL_TIME)
                if done:
                    self.job_metadata[job_id][f"decoder{int(decoder_block_id)}_num_of_tok_completed"] += num_tokens
                    self.job_metadata[job_id]["last_decoder_block_completion_time"] = self.GLOBAL_TIME
                    try:
                        self.job_metadata[job_id][f"decoder{int(decoder_block_id+1)}_take"] += num_tokens
                    except KeyError:
                        pass
                    
        if self.decoder_blocks[self.lowest_decoder_block_id].is_processing is False:
            job_id = self.request_open_job()

        earliest_job_iteration = None    

        for decoder_block_id in sorted(self.decoder_blocks.keys()):
            decoder_block = self.decoder_blocks[decoder_block_id]
            if decoder_block.is_processing is False:
                earliest_job_iteration = None
                earliest_time = float('inf')
                for job_iteration in self.current_open_jobs:
                    if self.job_metadata[job_iteration.job.job_id][f"decoder{decoder_block_id}_take"] > 0:
                        completion_time = self.job_metadata[job_iteration.job.job_id]["last_decoder_block_completion_time"]
                        if completion_time < earliest_time:
                            earliest_time = completion_time
                            earliest_job_iteration = job_iteration

                if earliest_job_iteration is not None and earliest_job_iteration:
                    decoder_block.start_processing(earliest_job_iteration, self.GLOBAL_TIME)
                    self.use_kv_memory(earliest_job_iteration.token_in_iteration, job_id=earliest_job_iteration.job.job_id)
                    self.job_metadata[earliest_job_iteration.job.job_id][f"decoder{decoder_block_id}_take"] = 0

    def find_completed_jobs_iterations(self):
        """
        Find and remove all completed jobs from the current_open_jobs queue.
        A job is considered complete if the number of tokens completed is equal to the number of tokens for all stages.
        """
        completed_jobs = []
        for job_iteration in self.current_open_jobs:
            job_id = job_iteration.job.job_id
            num_of_tok = self.job_metadata[job_id]["number_of_tokens"]
            completed = all(
            self.job_metadata[job_id][f"decoder{decoder_block_id}_num_of_tok_completed"] == num_of_tok
            for decoder_block_id in self.decoder_blocks
            )
            if completed:
                completed_jobs.append(job_iteration)

        for job in completed_jobs:
            self.current_open_jobs.remove(job)

        return completed_jobs


            

    def request_open_job(self):
        """
        Request a new job to be processed by the decoder block with the lowest index.
        """
        highest_weight_job_iteration = self.get_highest_weight_job()
        if highest_weight_job_iteration:
            self.current_open_jobs.append(highest_weight_job_iteration)
            self.job_metadata[highest_weight_job_iteration.job.job_id]["start_time"] = self.GLOBAL_TIME
            self.job_metadata[highest_weight_job_iteration.job.job_id][f"decoder{self.lowest_decoder_block_id}_take"] = highest_weight_job_iteration.token_in_iteration
            return highest_weight_job_iteration.job.job_id
        return None


    def finish_job_itertaion(self, job_iteration):
        """
        Finalize a job, log its metadata, and clean up its resources.
        """
        job_id = job_iteration.job.job_id
        self.job_metadata[job_id]["completion_time"] = self.GLOBAL_TIME
        self.job_metadata[job_id]["processing_time"] += self.GLOBAL_TIME - self.job_metadata[job_id]["start_time"]
        self.completed_jobs_log.append(self.job_metadata[job_id])
        self.outgoing_queues[job_id].append(job_iteration)  # Add to outgoing queue

    def push_outgoing_jobs(self):
        """
        Push jobs from outgoing queues to the next server or communication link.
        Only push if the next server and communication link are available (not dropped out).
        """
        for job_id, queue in self.outgoing_queues.items():
            while queue:
                job_iteration = queue.pop(0)
                if not job_iteration.job.job_id == job_id:
                    raise ValueError(f"Job ID mismatch: {job_iteration.job.job_id} != {job_id}")
                try:
                    next_server = job_iteration.job.get_next_server(self.server_id)  # Assume job contains routing info
                    comm_link = job_iteration.job.get_next_commLink(self.server_id)      # Assume job contains comm link info
                except AttributeError as e:
                    raise SystemExit(f"Error processing job {job_id}: {e}")



                # Check if next server and comm link are available
                if next_server and next_server.is_dropped_out is False and comm_link and comm_link.link_state == "UP": # inconsistent between comm link and server - revision
                    comm_link.add_job_iteration(job_iteration, self.GLOBAL_TIME)  # Forward the job
                    self.free_token_buffer_memory(job_iteration.token_in_iteration, job_id=job_id)
                else:
                    # If not available, requeue the job
                    self.outgoing_queues[job_id].append(job_iteration)
                    break


    def check_and_end_jobs_iteration(self):
        """
        Check if any jobs have completed all stages and end them.
        """
        completed_job_iterations = self.find_completed_jobs_iterations()
        for job_iterations in completed_job_iterations:
            self.finish_job_itertaion(job_iterations)
    # -------------------------------------------------------------------------
    #                        Server Step
    # -------------------------------------------------------------------------
    def run_time_step(self, time_quantum, current_time):
        """
        Perform all operations for a single global time step.
        1. Check/update dropout status.
        2. Update throughput if not dropped out.
        3. Update job weights.
        4. Process ongoing job or select the highest-weight job from the queues.
        5. Push completed jobs to outgoing queues.
        """
        self.GLOBAL_TIME = current_time  # Assume GLOBAL_TIME is provided by the System.
        self.GLOBAL_TIME_QUANTUM = time_quantum  # Assume GLOBAL_TIME_QUANTUM is provided by the System.
        # 1. Handle dropout
        self.check_for_dropout()

        # 2. Update throughput
        if not self.is_dropped_out:
            if (self.actual_throughput == 0) or self.check_start_of_server_slot():
                
                self.actual_throughput = self.sample_actual_throughput()
        else:
            self.actual_throughput = 0
            return

        # 3. Update weights
        
        self.update_queue_weights()

        # 4. Process ongoing job or select the highest-weight job
        self.process_time_step_at_decoder()

        # 5. Push outgoing jobs
        self.push_outgoing_jobs()

        # 6. Check and end jobs iteration
        self.check_and_end_jobs_iteration()

       # self.memory_use_alloc_sync()


    def start_job(self, job_iteration):
        """
        Start processing a job:
          - Allocate memory for it
          - Mark its start time
          - create incoming and outgoing queues fot the job
          - start job_meta data for the job
            - forward the announcment message to next server in routing row
        """
        job = job_iteration.job
        job_iteration.check_route_index(self.server_id)
        self.create_queues_and_trackinDict_for_job(job)
        self.allocate_job_memory(job)


        if job.status == "completed":
            self.terminate_job(job.job_id)

        self.outgoing_queues[job.job_id].append(job_iteration)


    def terminate_job(self, job_id):
        """
        Terminate a job:
          - Free its memory
          - Log its completion time
          - Update throughput
          - Update job queues
        """
        if job_id not in self.job_metadata:
            raise ValueError(f"Job {job_id} is not registered in metadata.")

        self.job_metadata[job_id]["job_end"] = True
        self.free_job_memory(job_id)
        self.release_job_memory(job_id)
        job = self.incoming_queues[job_id]["in_queue"].pop(0)
        self.outgoing_queues[job_id].append(job) 


    def clean_up_after_termination(self):    
        """
        Clean up after terminating a job:
          - Free memory
          - Update throughput
          - Update job queues
        """

        for job_id in list(self.job_metadata.keys()):
            if self.job_metadata[job_id]["job_end"] and not self.outgoing_queues[job_id]:
                self.remove_job_from_queues(job_id)
                del self.job_metadata[job_id]
        

    # -------------------------------------------------------------------------
    #                   memory service statistics
    # -------------------------------------------------------------------------  
    def rate_of_memory_usage(self):
        """
        Calculate the rate of memory usage.
        """
        return self.all_memory_used / self.all_memory_allocated

    def track_memory_usage(self):
        """
        Track memory usage for the server.
        """
        print(f"Memory usage analysis! {self.GLOBAL_TIME}")
        self.memory_usage_tracking[self.GLOBAL_TIME] = {
            
            "alloc_mem": self.all_memory_allocated,
            "used_mem": self.all_memory_used,
            "mem_ratio": self.rate_of_memory_usage(),
            "num_jobs": len(self.job_metadata.keys())
        }    
        return self.memory_usage_tracking

    def save_memory_usage_tracking(self, dir_path, filname_prefix, filename_suffix):
        """
        Save memory usage tracking to a csv-file.
        """
        filename = f"{filname_prefix}_{self.id}_{filename_suffix}.csv"
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "w") as file:
            file.write("time,alloc_mem,used_mem,mem_ratio,num_jobs\n")
            for time, data in self.memory_usage_tracking.items():
                file.write(f"{time},{data['alloc_mem']},{data['used_mem']},{data['mem_ratio']},{data['num_jobs']}\n")
