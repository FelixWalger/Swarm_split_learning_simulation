import time
import math
import numpy as np
# Monkey patch the method.
import sys
import os

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Assume the module is one level up from Test_scripts:
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path so Python can find DecoderBlock.py
sys.path.insert(0, project_root)
from DecoderBlock import DecoderBlock as DB  
# ---------------------------------------------------------------------------
# Dummy job class for testing DecoderBlock.
# ---------------------------------------------------------------------------
class DummyJob:
    def __init__(self, job_id, current_number_of_tokens):
        """
        A simple dummy job that has an id and a required number of tokens
        to process for a decoder block.
        """
        self.job_id = job_id
        self.current_number_of_tokens = current_number_of_tokens
        self.finished = False  # This flag is not used by DecoderBlock but might be used by Server




# ---------------------------------------------------------------------------
# Test Script
# ---------------------------------------------------------------------------
def main():
    # Simulation parameters.
    quantum = 0.1              # Simulation time quantum (seconds).
    actual_throughput = 5.0    # Tokens per second processed by the block.
    global_time = 0.0

    # Create a dummy job that requires a total of 20 tokens.
    dummy_job = DummyJob(job_id=1, current_number_of_tokens=20)

    # Create a DecoderBlock instance.
    decoder = DB(block_id=1, processing_efficiency=1.0)

    # Start processing the job.
    decoder.start_processing(dummy_job, global_time)
    global_time += quantum

    print("\nStarting simulation loop:")
    # Run the simulation loop until the job is fully processed.
    while decoder.is_processing:
        # Process tokens for this time quantum.

        job_id, done, processed = decoder.process_token(actual_throughput, quantum, global_time)

        # Increment the global simulation time.
        global_time += quantum
        print(f"Time {global_time:.2f} sec: Processed {processed} tokens; Total processed: {decoder.total_processed}/{decoder.required_tokens}; done={done}")

        # Sleep briefly to slow down output (optional).
        time.sleep(0.05)
    
    print(f"Time {global_time:.2f} sec: Processed {processed} tokens; Total processed: {decoder.total_processed}/{decoder.required_tokens}; done={done}")

    print("\nJob processing complete.")

if __name__ == "__main__":
    main()