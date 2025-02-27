import math
import os
import json

class DecoderBlock:
    """
    A simple decoder block that processes tokens for a job.
    
    Attributes:
        id (int): Unique identifier for the block.
        processing_efficiency (float): Factor to adjust throughput (default 1.0).
        is_processing (bool): Flag indicating if a job is being processed.
        current_job: The job currently being processed.
        accumulated_tokens (float): Accumulated fractional tokens processed (from multiple quanta).
        total_processed (int): Total number of tokens processed so far in this block for the current job.
        required_tokens (int): Total tokens required to finish this stage (from job.current_number_of_tokens).
        start_time (float): Global time when processing started.
    """
    
    def __init__(self, block_id, processing_efficiency=1.0, config_filename='config_simulation.json'):
        self.block_id = block_id
        self.processing_efficiency = processing_efficiency
        self.is_processing = False
        self.current_job = None
        self.accumulated_tokens = 0.0
        self.total_processed = 0
        self.required_tokens = 0
        self.start_time = None

        config_path = os.path.join(os.path.dirname(__file__), config_filename)
        with open(config_path, 'r') as config_file:
            config_simulation = json.load(config_file)
        self.precision = config_simulation["precision"]
        self.d_model = config_simulation["d_model"]

        print(f"num params: {self.d_model*4+4*self.d_model**2+8*self.d_model**2}")
        self.num_param = self.d_model*4+4*self.d_model**2+8*self.d_model**2

    def get_model_memory_usage(self):
        """
        Return the memory usage of the model in params.
        
        The memory usage is computed as the product of the precision and the model dimension.
        """
        return (self.d_model*4+4*self.d_model**2+8*self.d_model**2)

    def __int__(self):
       """Allow conversion to int to yield the block's block_id."""
       return self.block_id

    def __repr__(self):
        return (f"DecoderBlock(id={self.block_id}, is_processing={self.is_processing}, "
                f"total_processed={self.total_processed}/{self.required_tokens})")

    def start_processing(self, job_iteration, global_time):
        """
        Start processing a new job.
        
        Args:
            job: The job to process. It is expected that the job has attributes:
                 - job.job_id: A unique identifier.
                 - job.current_number_of_tokens: Total tokens to process for this stage.
            global_time (float): The current global simulation time.
        """
        self.current_job = job_iteration.job
        self.current_job_iteration = job_iteration
        self.is_processing = True
        self.accumulated_tokens = 0.0
        self.total_processed = 0
        # We assume that the job object provides the total number of tokens
        # required for this decoder block.
        self.required_tokens = job_iteration.token_in_iteration  
        self.start_time = global_time
        print(f"DecoderBlock {self.block_id} started processing job {self.current_job.job_id} at time {global_time}")

    def process_token(self, actual_throughput, quantum, global_time):
        """
        Process tokens during a simulation time quantum.
        
        The throughput provided (tokens per second) is multiplied by the quantum
        (in seconds) to determine how many tokens would be processed in that period.
        Because quantum is very small compared to the time to process a full token,
        fractional tokens are accumulated until at least one full token is ready.
        
        Args:
            actual_throughput (float): Token processing rate (tokens/second) for this block.
            quantum (float): The simulation time quantum (seconds).
            global_time (float): The current global simulation time.
        
        Returns:
            tuple: (job_id, done, num_tokens)
                - job_id: The id of the job being processed.
                - done (bool): True if the required number of tokens have been processed;
                               False otherwise.
                - num_tokens (int): The number of whole tokens processed during this call.
        """
        if not self.is_processing or self.current_job is None:
            return None, False, 0

        # Compute tokens processed during this time quantum.
        # Multiply by processing efficiency if needed.
        tokens_this_step = actual_throughput * quantum * self.processing_efficiency

        # Accumulate fractional tokens.
        self.accumulated_tokens += tokens_this_step

        # Determine how many whole tokens have been accumulated.
        processed = int(math.floor(self.accumulated_tokens))
        self.accumulated_tokens -= processed

        # Do not process more than what is required.
        remaining = self.required_tokens - self.total_processed
        if processed > remaining:
            processed = remaining

        self.total_processed += processed

        # Check if the decoder block has finished processing the job.
        done = self.total_processed >= self.required_tokens
        if done:
            self.is_processing = False
            print(f"DecoderBlock {self.block_id} finished processing job {self.current_job.job_id} at time {global_time}")

        return self.current_job.job_id, done, self.total_processed

