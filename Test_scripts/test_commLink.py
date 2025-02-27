import os
import json
import random
import math
import numpy as np
import time
import sys

# ======================================================================
# 1. Create a dummy configuration file if not already present.
# ======================================================================



# ======================================================================
# 2. Import the CommunicationLink class.
# (Assuming the CommunicationLink class code is in the same directory.)
# ======================================================================
# from CommunicationLink import CommunicationLink
# For this test script, we assume the CommunicationLink class is defined above.
# (If it is in a separate module, uncomment the line above.)

# ======================================================================
# 3. Define dummy entity and job classes for testing.
# ======================================================================
class DummyEntity:
    """A simple dummy for sender/receiver with the minimal interface."""
    def __init__(self, name):
        self.name = name

    def start_job(self, job):
        print(f"[{self.name}] Starting job {job.job_id} (status: {job.status}).")

    def add_jobIteration_to_queue(self, job_iteration):
        print(f"[{self.name}] Enqueuing JobIteration {job_iteration.iteration_id} for job {job_iteration.job.job_id}.")


class DummyJob:
    """A dummy job that carries an ID, token count, and a status."""
    def __init__(self, job_id, tokens, status="initialization"):
        self.job_id = job_id
        self.tokens = tokens  # Total tokens per job (used to compute bits)
        self.status = status  # e.g., "initialization", "processing", etc.


class DummyJobIteration:
    """A dummy job iteration that holds a reference to a job and the number of tokens to process."""
    def __init__(self, iteration_id, job, token_in_iteration):
        self.iteration_id = iteration_id
        self.job = job
        self.token_in_iteration = token_in_iteration  # Number of tokens in this iteration


# ======================================================================
# 4. Monkey-patch CommunicationLink._deliver_completed_transmissions to fix a bug.
# ======================================================================
def fixed_deliver_completed_transmissions(self, current_time):
    """
    Corrected delivery method that uses the local variable `job_iter` instead of
    the erroneous `self.job_iter`.
    """
    completed = []
    for tx in self.active_transmissions:
        arr_t = tx["arrival_time_at_receiver"]
        if arr_t is not None and current_time >= arr_t:
            completed.append(tx)
    for tx in completed:
        job_iter = tx["job_iteration"]
        print(f"[Link {self.link_id}] JobIteration {job_iter.iteration_id} arrived at {self.to_entity.name} "
              f"at time={current_time:.2f} (trans_start={tx['transmission_start_time']:.2f}, "
              f"end={tx['transmission_end_time']:.2f}, arrival={tx['arrival_time_at_receiver']:.2f})")
        if job_iter.job.status == "initialization":
            self.to_entity.start_job(job_iter.job)
        else:
            self.to_entity.add_jobIteration_to_queue(job_iter)
    self.active_transmissions = [tx for tx in self.active_transmissions if tx not in completed]

# Monkey patch the method.
import sys
import os

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Assume the module is one level up from Test_scripts:
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add the project root to sys.path so Python can find Communicationlink.py
sys.path.insert(0, project_root)
from Communicationlink import CommunicationLink as CL  # If CommunicationLink is in its own module.
# Otherwise, if the class is defined in this file, simply assign:
CL._deliver_completed_transmissions = fixed_deliver_completed_transmissions

# If the class is defined in the current namespace (for testing), do:
# CommunicationLink._deliver_completed_transmissions = fixed_deliver_completed_transmissions


# ======================================================================
# 5. Create dummy sender and receiver entities.
# ======================================================================
sender = DummyEntity("Sender")
receiver = DummyEntity("Receiver")


# ======================================================================
# 6. Instantiate the CommunicationLink.
# ======================================================================
# For this test, we choose a time_quantum of 1 second.
time_quantum = 1.0
link = CL(
    link_id=1,
    from_entity=sender,
    to_entity=receiver,
    time_quantum=time_quantum,
    initial_state="UP"
)

# ======================================================================
# 7. Create a dummy job and a job iteration.
# ======================================================================
dummy_job = DummyJob(job_id=101, tokens=150, status="initialization")
dummy_job_iter = DummyJobIteration(iteration_id=1, job=dummy_job, token_in_iteration=1)
dummy_job1 = DummyJob(job_id=102, tokens=100, status="decoding")
dummy_job_iter1 = DummyJobIteration(iteration_id=2, job=dummy_job, token_in_iteration=1000)

# ======================================================================
# 8. Enqueue the job iteration for transmission.
# ======================================================================
current_sim_time = 0.0
print("\n--- Adding JobIteration to link ---")
link.add_job_iteration(dummy_job_iter, current_sim_time)
link.add_job_iteration(dummy_job_iter1, current_sim_time)

# ======================================================================
# 9. Simulate a series of time steps.
# ======================================================================
total_simulation_time = 20  # seconds
print("\n--- Starting simulation steps ---")
while current_sim_time <= total_simulation_time:
    print(f"\n[Time {current_sim_time:.2f} sec] Link state: {link.link_state}, "
          f"Time to next switch: {link.time_to_next_state_switch:.2f} sec")
    link.process_time_step(current_sim_time)
    current_sim_time += time_quantum
    # (In a real simulation you might wait for real time or simply advance the simulation clock.)
    time.sleep(0.1)  # Short sleep to slow down output for demonstration purposes

# ======================================================================
# 10. Test some helper functions.
# ======================================================================
dummy_bits_per_token = dummy_job.tokens * link.precision * link.d_model
expected_latency = link.get_expected_latency_per_token(dummy_bits_per_token)
print(f"\nExpected one-way latency per token (best-case) for dummy_job_iter: {expected_latency:.4f} seconds")

dummy_bits_per_token1 = dummy_job1.tokens * link.precision * link.d_model
expected_latency1 = link.get_expected_latency_per_token(dummy_bits_per_token1)
print(f"Expected one-way latency per token (best-case) for dummy_job_iter1: {expected_latency1:.4f} seconds")

print(f"Link down rate: {link.get_down_rate():.4f} 1/sec")
print(f"Link up rate: {link.get_up_rate():.4f} 1/sec")