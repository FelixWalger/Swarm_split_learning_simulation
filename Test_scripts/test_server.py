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

# =============================================================================
# Create a dummy configuration file if not present.
# =============================================================================
CONFIG_FILENAME = 'config_simulation.json'
if not os.path.exists(CONFIG_FILENAME):
    dummy_config = {
        "server_properties": {
            "memory_range": [1000, 2000],            # Example memory capacity range (in bytes)
            "dropout_prob_range": [0.1, 0.1],
            "up_server_time_range_min": [1, 1],
            "down_server_time_range_min": [1, 1],
            "throughput_range": [50, 100],             # Throughput range (tokens/sec)
            "degradation_rate": [0.01, 0.01]
        },
        "precision": 32,
        "d_model": 512
    }
    with open(CONFIG_FILENAME, "w") as f:
        json.dump(dummy_config, f, indent=2)
    print(f"Created dummy config file '{CONFIG_FILENAME}'.")

# =============================================================================
# Dummy Job and JobIteration classes for testing
# =============================================================================
class DummyJob:
    def __init__(self, job_id, token_buffer_alloc, KV_cache_alloc):
        """
        A dummy job for testing. It provides:
          - job_id: Unique identifier.
          - current_number_of_tokens: Total tokens to process.
          - token_buffer_alloc: Amount of memory (in bytes) to allocate for the token buffer.
          - KV_cache_alloc: Amount of memory (in bytes) to allocate for the KV cache.
        """
        self.job_id = job_id
        self.all_tokens_sum = 0
        self.token_buffer_alloc = token_buffer_alloc
        self.KV_cache_alloc = KV_cache_alloc
        self.finished = False  # May be used by Server later.
        # Dummy routing info (empty for this test).
        self.routing_info = {}

    def __repr__(self):
        return (f"DummyJob(id={self.job_id}, tokens={self.all_tokens_sum}, "
                f"token_buf_alloc={self.token_buffer_alloc}, KV_cache_alloc={self.KV_cache_alloc})")

class DummyJobIteration:
    def __init__(self, iteration_id, job, token_in_iteration):
        """
        A dummy job iteration wraps a job and indicates how many tokens are processed
        in this iteration.
        """
        self.iteration_id = iteration_id
        self.job = job
        self.token_in_iteration = token_in_iteration

# =============================================================================
# DecoderBlock class (as defined earlier)
# =============================================================================
# =============================================================================
# Assume the Server class is defined as provided earlier.
# (For this test script, the complete Server class code is assumed available.)
# =============================================================================
# For example, if the Server class is defined in a module, you might do:
# from Server import Server
# Here we assume it is defined in the current namespace.

# =============================================================================
# Test Functions
# =============================================================================

def test_single_decoder_block():
    print("=== Test: Single Decoder Block Integration ===")
    # Create a dummy job.
    job = DummyJob(job_id=1, token_buffer_alloc=256, KV_cache_alloc=512)
    # Create a dummy job iteration.
    job_iter = DummyJobIteration(iteration_id=1, job=job, token_in_iteration=10)


    # Create a single decoder block.
    decoder = DB(block_id=1, processing_efficiency=1.0)
    print(f"Created decoder block: {decoder}")
    decoder_list = [decoder]
    print(f"Decoder list: {decoder_list}")
    
    # Instantiate the Server with one decoder block.
    server = Server(server_id=1, decoder_blocks=decoder_list, base_throughput=1)
    print("\nMemory usage tracking1.1:")
    print(server.track_memory_usage())

    # --- Monitor initial memory state ---
    print("\nInitial memory allocation:")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"Model param allocated: {server.model_param_mem_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")

    server.track_memory_usage()
    print("\nMemory usage tracking1.2:")
    print(server.memory_usage_tracking)    
    # Start the job on the server (which allocates memory and creates queues).
    server.start_job(job_iter)
    server.add_jobIteration_to_queue(job_iter)
    
    print("\nAfter starting job (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    
    # --- Process the job using the single decoder block ---
    quantum = 0.1                # Small time quantum.
    global_time = 0.0
    actual_throughput = server.actual_throughput  # Tokens per second.
    print(f"\nStarting simulation loop with actual throughput {actual_throughput}:")
    global_time += quantum
    server.run_time_step(quantum, global_time)
    global_time += quantum
    server.run_time_step(quantum, global_time)
    global_time += quantum
    print(f"======================== decoder is processing: {decoder.is_processing}")
    print("\nAfter starting job (memory USE):")
    print(f"KV use: {server.kv_mem_used}")
    print(f"Token buffer use: {server.token_buffer_used}")
    print(f"All memory use: {server.all_memory_used}")
    print("\nProcessing job on single decoder block:")
    while decoder.is_processing:
        server.run_time_step(quantum, global_time)
        #print(f"Time {global_time:.2f} sec: Processed {processed} tokens; Total processed: {decoder.total_processed}/{decoder.required_tokens}")
        print(f"\n ----------------- Memory use at {global_time}:")
        print(f"KV use: {server.kv_mem_used}")
        print(f"Token buffer use: {server.token_buffer_used}")
        print(f"All memory use: {server.all_memory_used}")
        global_time += quantum
        time.sleep(0.05)
    
    server.run_time_step(quantum, global_time)
    global_time += quantum

    server.run_time_step(quantum, global_time)
    global_time += quantum
    print("\n After ending job before free (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    # --- Free and release job memory ---
    print("\nFreeing job memory:")
    # release_job_memory returns a tuple; we extract the first two values.
   
    print("\nAfter  job end  before free (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")

    server.memory_use_alloc_sync()
    server.track_memory_usage()
    print("\nMemory usage tracking:")
    print(server.memory_usage_tracking)
    print("=== End of Single Decoder Block Test ===\n")


def test_multiple_decoder_blocks():
    print("=== Test: Multiple Decoder Blocks Integration ===")
    # Create a dummy job.
    job = DummyJob(job_id=2, token_buffer_alloc=150, KV_cache_alloc=300)
    job_iter = DummyJobIteration(iteration_id=1, job=job, token_in_iteration=5)
    
    # Create multiple decoder blocks.
    decoder1 = DB(block_id=1, processing_efficiency=1.0)
    decoder2 = DB(block_id=2, processing_efficiency=1.0)
    decoder_list = [decoder1, decoder2]
    
    # Instantiate the Server with multiple decoder blocks.
    server = Server(server_id=2, decoder_blocks=decoder_list, base_throughput=1)
    
    print("\nMemory usage tracking2.1:")
    print(server.track_memory_usage())
    
    # --- Monitor initial memory state ---
    print("\nInitial memory allocation:")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"Model param allocated: {server.model_param_mem_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    
    server.track_memory_usage()
    print("\nMemory usage tracking2.2:")
    print(server.memory_usage_tracking)
    
    # Start the job on the server (which allocates memory and creates queues).
    server.start_job(job_iter)
    server.add_jobIteration_to_queue(job_iter)
    
    print("\nAfter starting job (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    
    # --- Process the job using multiple decoder blocks ---
    quantum = 0.1
    global_time = 0.0
    actual_throughput = server.actual_throughput
    
    print(f"\nStarting simulation loop with actual throughput {actual_throughput}:")
    global_time += quantum
    server.run_time_step(quantum, global_time)
    global_time += quantum
    server.run_time_step(quantum, global_time)
    global_time += quantum
    print(f"======================== decoder1 is processing: {decoder1.is_processing}")
    print("\nAfter starting job (memory USE):")
    print(f"KV use: {server.kv_mem_used}")
    print(f"Token buffer use: {server.token_buffer_used}")
    print(f"All memory use: {server.all_memory_used}")
    print("\nProcessing job on multiple decoder blocks:")
    
    while decoder1.is_processing or decoder2.is_processing:
        server.run_time_step(quantum, global_time)
        print(f"\n ----------------- Memory use at {global_time}:")
        print(f"KV use: {server.kv_mem_used}")
        print(f"Token buffer use: {server.token_buffer_used}")
        print(f"All memory use: {server.all_memory_used}")
        global_time += quantum
        time.sleep(0.05)
    
    server.run_time_step(quantum, global_time)
    global_time += quantum

    server.run_time_step(quantum, global_time)
    global_time += quantum
    print("\n After ending job before free (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    
    # --- Free and release job memory ---
    print("\nFreeing job memory:")
    print("\nAfter job end before free (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    
    server.memory_use_alloc_sync()
    server.track_memory_usage()
    print("\nMemory usage tracking:")
    print(server.memory_usage_tracking)
    print("=== End of Multiple Decoder Blocks Test ===\n")

def test_multiple_jobs():
    print("=== Test: Multiple Jobs Integration ===")
    
    # Create multiple dummy jobs and job iterations.
    jobs = [
        DummyJob(job_id=1, token_buffer_alloc=256, KV_cache_alloc=512),
        DummyJob(job_id=2, token_buffer_alloc=150, KV_cache_alloc=300),
        DummyJob(job_id=3, token_buffer_alloc=200, KV_cache_alloc=400)
    ]
    
    job_iterations = [
        DummyJobIteration(iteration_id=1, job=jobs[0], token_in_iteration=10),
        DummyJobIteration(iteration_id=2, job=jobs[1], token_in_iteration=15),
        DummyJobIteration(iteration_id=3, job=jobs[2], token_in_iteration=20)
    ]
    
    # Create multiple decoder blocks.
    decoder1 = DB(block_id=1, processing_efficiency=1.0)
    decoder2 = DB(block_id=2, processing_efficiency=0.9)
    decoder_list = [decoder1, decoder2]
    
    # Instantiate the Server with multiple decoder blocks.
    server = Server(server_id=3, decoder_blocks=decoder_list, base_throughput=1)
    
    print("\nMemory usage tracking for multiple jobs:")
    print(server.track_memory_usage())
    
    # Start the jobs on the server (which allocates memory and creates queues).
    for job_iter in job_iterations:
        server.start_job(job_iter)
        server.add_jobIteration_to_queue(job_iter)
    
    print("\nAfter starting jobs (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    
    # --- Process the jobs using multiple decoder blocks ---
    quantum = 0.1
    global_time = 0.0
    actual_throughput = server.actual_throughput
    
    print(f"\nStarting simulation loop with actual throughput {actual_throughput}:")
    while any(decoder.is_processing for decoder in decoder_list):
        server.run_time_step(quantum, global_time)
        print(f"\n ----------------- Memory use at {global_time}:")
        print(f"KV use: {server.kv_mem_used}")
        print(f"Token buffer use: {server.token_buffer_used}")
        print(f"All memory use: {server.all_memory_used}")
        global_time += quantum
        time.sleep(0.05)
    
    print("\nAfter ending jobs (memory allocation):")
    print(f"KV allocated: {server.kv_mem_allocated}")
    print(f"Token buffer allocated: {server.token_buffer_allocated}")
    print(f"All memory allocated: {server.all_memory_allocated}")
    
    # --- Free and release job memory ---
    print("\nFreeing job memory:")
    server.memory_use_alloc_sync()
    server.track_memory_usage()
    print("\nMemory usage tracking:")
    print(server.memory_usage_tracking)
    print("=== End of Multiple Jobs Test ===\n")

def main():
    test_single_decoder_block()
    test_multiple_decoder_blocks()
    test_multiple_jobs()

if __name__ == "__main__":
    main()