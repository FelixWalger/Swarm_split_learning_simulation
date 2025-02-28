import random
import math
import numpy as np
import json
import os

class CommunicationLink:
    """
    A communication link that can simultaneously transmit multiple JobIteration objects
    subject to a total data rate limit. Each transmission is advanced every simulation
    step until its bits are fully sent. After that, the data "propagates" for some time
    (propagation_delay + jitter) before reaching the receiving entity.

    The link can be either UP or DOWN, modeled by a continuous-time Markov chain:
      - UP -> DOWN with rate = recovery_rate
      - DOWN -> UP with rate = failure_rate
    When DOWN, no transmission progress is possible until the link comes back UP.
    """

    def __init__(self,
                 link_id,
                 from_entity,
                 to_entity,
                 data_rate_bps=None,
                 speed_of_signal_m_s=None,
                 physical_length_m=None,
                 jitter_logN_mu=None,
                 jitter_logN_sigma=None,
                 time_quantum=1.0,
                 recovery_rate=None,
                 failure_rate=None,
                 initial_state="UP",
                 config_filename='config_simulation.json'
                 ):
        """
        :param link_id:              Unique identifier for the link.
        :param from_entity:          The sender (Server or Client).
        :param to_entity:            The receiver (Server or Client).
        :param data_rate_bps:        Total link capacity in bits/second (shared by all transmissions).
        :param speed_of_signal_m_s:  Speed of signal (m/s). Default ~ 2e8.
        :param physical_length_m:    Link length (meters).
        :param jitter_logN_mu:       Mu parameter for lognormal jitter (sec).
        :param jitter_logN_sigma:    Sigma parameter for lognormal jitter (sec).
        :param time_quantum:         Simulation time quantum (sec).
        :param recovery_rate:            Rate (1/sec) at which the link transitions from UP -> DOWN.
        :param failure_rate:              Rate (1/sec) at which the link transitions from DOWN -> UP.
        :param initial_state:        "UP" or "DOWN" for the starting state.
        """



        # Track ongoing transmissions
        self.active_transmissions = []  # List[dict]
        # Load configuration from JSON file
        config_path = os.path.join(os.path.dirname(__file__), config_filename)
        with open(config_path, 'r') as config_file:
            config_simulation = json.load(config_file)

        network_props = config_simulation["network_properties"]
        self.precision = config_simulation["precision"]
        self.d_model = config_simulation["d_model"]


        self.link_id = link_id
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.data_rate_bps = data_rate_bps if data_rate_bps is not None else random.uniform(*network_props["data_rate_range_Mbits/s"]) * 1e6
        self.speed_of_signal_m_s = speed_of_signal_m_s if speed_of_signal_m_s is not None else network_props["propagation_speed_km/s"] * 1000
        self.physical_length_m = physical_length_m if physical_length_m is not None else random.uniform(*network_props["distance_range_km"]) * 1000
        if jitter_logN_mu is None or jitter_logN_sigma is None:
            jitter_params = random.choice(network_props["jitter_lognorm_param_tuples_mu_sigma"])
            self.jitter_logN_mu, self.jitter_logN_sigma = jitter_params
            print(f"Jitter parameters selected: mu={self.jitter_logN_mu}, sigma={self.jitter_logN_sigma}")
        else:
            self.jitter_logN_mu = jitter_logN_mu
            self.jitter_logN_sigma = jitter_logN_sigma
        self.time_quantum = time_quantum
        print(f"down_link_time_range_min: {network_props['down_link_time_range_min']}")
        self.recovery_rate = recovery_rate if recovery_rate is not None else 1.0 / random.uniform(*network_props["down_link_time_range_min"]) / 60
        print(f"Down rate: {self.recovery_rate}")
        self.failure_rate = failure_rate if failure_rate is not None else 1.0 / random.uniform(*network_props["up_link_time_range_min"]) / 60
        self.link_state = initial_state.upper()  # "UP" or "DOWN"

        # Time (in seconds) until next state transition
        if self.link_state == "UP":
            # sample the time to drop
            self.time_to_next_state_switch = np.random.exponential(self.failure_rate) if self.recovery_rate > 0 else float('inf')
        else:
            # sample the time to recover
            self.time_to_next_state_switch = np.random.exponential(self.recovery_rate) if self.failure_rate > 0 else float('inf')



        # Track ongoing transmissions
        self.active_transmissions = []  # List[dict]

    @classmethod
    def from_existing_comm_link(cls, existing_link, config_filename=None):
        """
        Create a new CommunicationLink instance by copying an existing one,
        with an option to use a different configuration file.
        """
        return cls(
            link_id=existing_link.link_id,
            from_entity=existing_link.from_entity,
            to_entity=existing_link.to_entity,
            data_rate_bps=existing_link.data_rate_bps,
            speed_of_signal_m_s=existing_link.speed_of_signal_m_s,
            physical_length_m=existing_link.physical_length_m,
            jitter_logN_mu=existing_link.jitter_logN_mu,
            jitter_logN_sigma=existing_link.jitter_logN_sigma,
            time_quantum=existing_link.time_quantum,
            recovery_rate=existing_link.recovery_rate,
            failure_rate=existing_link.failure_rate,
            initial_state=existing_link.link_state,
            config_filename=config_filename if config_filename is not None else existing_link.config_filename
        )
    def is_functioning(self):
        """
        Return True if the link is currently up (functioning),
        or False if it is in a down state (dropout).
        """
        return (self.link_state == "UP")

    # -------------------------------------------------------------------------
    #             Markov Chain Dropout Logic
    # -------------------------------------------------------------------------
    def _update_link_state(self):
        """
        Update the link's dropout state based on the continuous-time Markov chain.
        Subtract time_quantum from time_to_next_state_switch; if it goes below 0,
        switch state and sample the next sojourn time from the appropriate exponential distribution.
        """
        # If no transitions are possible (e.g., rates = 0), skip
        if (self.link_state == "UP" and self.recovery_rate <= 0) or \
           (self.link_state == "DOWN" and self.failure_rate <= 0):
            return

        # Decrement the timer for the current state
        self.time_to_next_state_switch -= self.time_quantum

        # If we haven't yet reached transition time, do nothing more
        if self.time_to_next_state_switch > 0:
            return

        # Otherwise, we've crossed a state boundary
        if self.link_state == "UP":
            # Switch to DOWN
            self.link_state = "DOWN"
            # Now sample how long we'll remain DOWN
            if self.failure_rate > 0:
                self.time_to_next_state_switch = np.random.exponential(self.recovery_rate)
            else:
                self.time_to_next_state_switch = float('inf')
            print(f"[Link {self.link_id}] Transitioned to DOWN state")

        else:  # was DOWN
            self.link_state = "UP"
            # Sample how long we'll remain UP
            if self.recovery_rate > 0:
                self.time_to_next_state_switch = np.random.exponential(1.0 / self.failure_rate)
            else:
                self.time_to_next_state_switch = float('inf')
            print(f"[Link {self.link_id}] Transitioned to UP state")

    # -------------------------------------------------------------------------
    #             Transmission Handling
    # -------------------------------------------------------------------------
    def add_job_iteration(self, job_iteration, current_time):
        """
        Initiate transmission of a new job_iteration.
        (No immediate dropout check, because dropout is handled
         by the Markov chain: if link is DOWN, no progress occurs.)
        """
        bits_to_send = self._compute_bits_for_job_iteration(job_iteration)

        propagation_delay = self.physical_length_m / self.speed_of_signal_m_s
        transmission_delay = bits_to_send / self.data_rate_bps
        jitter = (np.random.lognormal(self.jitter_logN_mu, self.jitter_logN_sigma) / 2) / 1000

        new_tx = {
            "job_iteration": job_iteration,
            "bits_remaining": bits_to_send,
            "propagation_delay": propagation_delay,
            "jitter": jitter,
            "transmission_start_time": current_time,
            "transmission_end_time": None,       # set when bits_remaining -> 0
            "arrival_time_at_receiver": None     # set after we finish bits
        }
        self.active_transmissions.append(new_tx)

        transmission_delay = bits_to_send / self.data_rate_bps
        print(f"[Link {self.link_id}] Enqueued JobIteration {job_iteration.iteration_id} "
              f"with {bits_to_send:.0f} bits to send. (transmission_delay={transmission_delay:.6f}s, prop_delay={propagation_delay:.6f}s, jitter~{jitter:.6f}s)")

    def process_time_step(self, time_quantum, current_time):
        """
        1) Update the link's dropout state (Markov chain).
        2) If link is UP, advance all active transmissions by time_quantum in a fair-share manner.
        3) Check for finished transmissions and deliver them if propagation is done.
        """
        self._update_link_state()  # step the Markov chain

        # If link is DOWN, no transmission progress is made, 
        # but we can still deliver if propagation is done.
        if self.link_state == "DOWN":
            self._deliver_completed_transmissions(current_time)
            return

        # Link is UP => proceed with transmissions
        if not self.active_transmissions:
            return  # nothing to send

        # Separate transmissions that are still sending bits
        transmissions_in_flight = [tx for tx in self.active_transmissions if tx["bits_remaining"] > 0]
        if transmissions_in_flight:
            total_capacity_bits = self.data_rate_bps * time_quantum
            total_demand_bits = sum(tx["bits_remaining"] for tx in transmissions_in_flight)

            if total_demand_bits <= total_capacity_bits:
                # Everyone finishes this step
                quantum_quantil_transmission=0
                for tx in transmissions_in_flight:
                    quantum_quantil_transmission += tx["bits_remaining"] / self.data_rate_bps
                    tx["bits_remaining"] = 0
                    tx["transmission_end_time"] = current_time-time_quantum+quantum_quantil_transmission #In a time step we transfer from state current_time = current_time + time quantum. The defintion /convention is that the fucntion parameter current_time is already the updated current_time
            else:
                # Fair share
                quantum_quantil_transmission = 0
                for tx in transmissions_in_flight:
                    fraction = tx["bits_remaining"] / total_demand_bits
                    bits_for_this_tx = total_capacity_bits * fraction
                    quantum_quantil_transmission =+ bits_for_this_tx/self.data_rate_bps
                    tx["bits_remaining"] = max(0, tx["bits_remaining"] - bits_for_this_tx)
                    if tx["bits_remaining"] == 0:
                        tx["transmission_end_time"] = current_time-time_quantum+quantum_quantil_transmission #In a time step we transfer from state current_time = current_time + time quantum. The defintion /convention is that the fucntion parameter current_time is already the updated current_time

            # Assign arrival_time_at_receiver for newly completed transmissions
            for tx in transmissions_in_flight:
                if tx["bits_remaining"] == 0 and tx["arrival_time_at_receiver"] is None:
                    end_t = tx["transmission_end_time"] or current_time
                    tx["arrival_time_at_receiver"] = end_t + tx["propagation_delay"] + tx["jitter"]

        # Attempt to deliver transmissions that have finished prop0agation
        self._deliver_completed_transmissions(current_time)

    def _deliver_completed_transmissions(self, current_time):
        completed = []
        for tx in self.active_transmissions:
            arr_t = tx["arrival_time_at_receiver"]
            if arr_t is not None and current_time >= arr_t:
                completed.append(tx)

        # Hand them over
        for tx in completed:
            job_iter = tx["job_iteration"]

            trans_end_time = tx['transmission_end_time'] if tx['transmission_end_time'] is not None else float('nan')

            arrival_time = arr_t if arr_t is not None else float('nan')
            if tx["transmission_end_time"] is float('nan') or arr_t is float('nan'):
                print("nan")
            #print(f"[Link {self.link_id}] JobIteration {job_iter.iteration_id} arrived at {self.to_entity} "
            #      f"at time={current_time:.2f} (trans_start={tx['transmission_start_time']:.2f}, "
             #     f"end={trans_end_time:.2f}, arrival={arrival_time:.2f})")
            if job_iter.job.status == "initialization":
                self.to_entity.start_job(job_iter)
            else:
                self.to_entity.add_jobIteration_to_queue(job_iter, current_time)   


        self.active_transmissions = [tx for tx in self.active_transmissions if tx not in completed]

    # -------------------------------------------------------------------------
    #                  Helper Functions
    # -------------------------------------------------------------------------
    def _compute_bits_for_job_iteration(self, job_iteration):
        tokens = self._get_job_iteration_token_count(job_iteration)
        bits_per_token = job_iteration.token_in_iteration * self.precision*self.d_model
        return tokens * bits_per_token if tokens != 0 else 50 * 8

    def _get_job_iteration_token_count(self, job_iteration):
        if hasattr(job_iteration, 'token_in_iteration'):
            return job_iteration.token_in_iteration
        return (1/self.d_model) 

    def get_expected_latency_per_token(self, bits_per_token=None):
        """
        Calculate the expected (average) one-way latency per token, ignoring concurrency effects
        and ignoring Markov chain downtime. This is a best-case scenario if the link were UP
        100% of the time and unshared.
        """
        if bits_per_token is None:
            bits_per_token = self.precision * self.d_model   # Default bits per token calculation
        if self.link_state == "DOWN":
            return 1e12  # Return a very high value instead of infinity
        transmission_time = bits_per_token / self.data_rate_bps
        propagation_delay = self.physical_length_m / self.speed_of_signal_m_s
        expected_jitter = expected_jitter = math.exp(self.jitter_logN_mu + 0.5 * (self.jitter_logN_sigma ** 2)) / 1000
        return transmission_time + propagation_delay + expected_jitter
    
    def get_recovery_rate(self):
        """
        Get the current down rate for the link.
        :return: Current down rate (1/sec).
        """
        return self.recovery_rate

    def get_failure_rate(self):
        """
        Get the current up rate for the link.
        :return: Current up rate (1/sec).
        """
        return self.failure_rate
