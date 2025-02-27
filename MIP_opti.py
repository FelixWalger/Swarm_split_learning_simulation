import itertools
from functools import reduce
from Server import Server
from Communicationlink import CommunicationLink
from DecoderBlock import DecoderBlock
import time

class OptiProblem:
    def __init__(self, servers, comm_links, max_L, num_decoder_blocks, SL_max, d_model, precision, epsilon = None):
        """
        Initialize the optimization problem.

        Parameters:
        - servers: List of Server objects representing S.
        - comm_links: List of CommLink objects representing (s, s') pairs.
        - max_L: Maximum number of decoder blocks (upper limit for L).
        """
        self.servers = servers
        self.comm_links = comm_links
        self.max_L = max_L
        self.best_solution = None
        self.num_decoder_blocks = num_decoder_blocks
        self.best_latency = float('inf')
        self.X = [[0 for _ in range(self.max_L)] for _ in range(len(self.servers))]
        self.Y = [[0 for _ in range(self.num_decoder_blocks)] for _ in range(self.max_L)]
        self.Z = [[0 for _ in range(len(self.servers))] for _ in range(len(self.servers))]
        self.exemplary_decoder_blocks = [DecoderBlock(i) for i in range(self.num_decoder_blocks)]
        self.SL_max = SL_max
        self.d_model = d_model
        self.precision = precision if precision is not None else 1e-5
        self.epsilon = epsilon

        #Logicial checks
        if self.max_L >= len(self.servers):
            raise ValueError("max_L cannot be greater than the number of servers.")
        if self.max_L >= self.num_decoder_blocks :
            raise ValueError("num_decoder_blocks cannot be greater than the number of servers.")



        self.X_dict = {}
        for s in range(len(self.servers)):
            for l in range(self.max_L):
                key = int(f"{s}{l}")
                self.X_dict[key] = {
                    "server_id": self.servers[s].server_id,
                    "object": self.servers[s],
                    "server_pos": s,
                    "level_pos": l
                }
          
          
        self.Z_dict = {}
        for s1 in range(len(self.servers)):
            for s2 in range(len(self.servers)):
                if s1 != s2:
                    key = int(f"{s1}{s2}")
                    comm_link_obj = self.get_comm_link(s1, s2)
                    self.Z_dict[key] = {
                        "comm_link_id": comm_link_obj.link_id,
                        "object": comm_link_obj,
                        "from_server_pos": s1,
                        "to_server_pos": s2
                    }
   
       


    def get_comm_link(self, from_server_pos, to_server_pos):
        from_key = int(f"{from_server_pos}{1}")
        to_key = int(f"{to_server_pos}{1}")
        from_server_obj = self.X_dict[from_key]["object"]
        to_server_obj = self.X_dict[to_key]["object"]
        for comm_link in self.comm_links:
            if (comm_link.from_entity == from_server_obj and comm_link.to_entity == to_server_obj):
                return comm_link
        raise ValueError(f"No communication link found between servers {from_server_obj.server_id} (position {from_server_pos}) and {to_server_obj.server_id} (position {to_server_pos})")

    def generate_solutions(self):
        """
        Generate all possible solutions for X, Y, and Z.

        Returns:
        - List of all valid (X, Y, Z) configurations.
        """
        print("Start creating solutions")
        solutions = []
        for X in itertools.product([0, 1], repeat=len(self.servers) * self.max_L):
            X = [X[i:i + self.max_L] for i in range(0, len(X), self.max_L)]
            for Y in itertools.product([0, 1], repeat=self.max_L * self.num_decoder_blocks):
                Y = [Y[i:i + self.num_decoder_blocks] for i in range(0, len(Y), self.num_decoder_blocks)]
                Z = [[0 for _ in range(len(self.servers))] for _ in range(len(self.servers))]
                for l in range(1, self.max_L):
                    for s in range(len(self.servers)):
                        for s_prime in range(len(self.servers)):
                            if X[s][l-1] == 1 and X[s_prime][l] == 1:
                                Z[s][s_prime] = 1
                solutions.append((X, Y, Z))
        print(f"Number of generated solutions: {len(solutions)}")
        return solutions
    
     # ------------------------------------------------------------------------
    # 1) Partition blocks into L consecutive subsets (for building Y)
    # ------------------------------------------------------------------------
    def partition_blocks_into_L_consecutive_subsets(self, num_blocks, max_L):
        """
        Partition the range [0..num_blocks-1] into max_L consecutive non-empty segments.
        Returns a list of such partitions, each partition is a list [(start, end), (start, end), ...].
        """
        results = []
        def backtrack(start, cuts_left, used_cuts):
            # If no more cuts left, take [start..num_blocks-1] as final segment
            if cuts_left == 0:
                if start < num_blocks:
                    used_cuts.append((start, num_blocks - 1))
                    results.append(used_cuts[:])
                    used_cuts.pop()
                return
            # Try placing a cut in [start+1 .. num_blocks - cuts_left]
            for cut_pos in range(start+1, num_blocks - cuts_left + 1):
                used_cuts.append((start, cut_pos - 1))
                backtrack(cut_pos, cuts_left - 1, used_cuts)
                used_cuts.pop()
        backtrack(0, max_L - 1, [])
        return results

    # ------------------------------------------------------------------------
    # 2) Memory-check function (already given; references self.Y)
    # ------------------------------------------------------------------------
    def check_basic_memory_constraint(self, s, l):
        """
        True if server 's' at level 'l' can store all decoder blocks assigned to level 'l'.
        This uses self.Y[l][d] to see which blocks are in that level.
        """
        assigned_decoder_blocks = [d for d in range(self.num_decoder_blocks) if self.Y[l][d] == 1]
        num_param_sum = sum(self.exemplary_decoder_blocks[d].num_param for d in assigned_decoder_blocks)

        server_capacity = self.X_dict[int(f"{s}{l}")]["object"].cache_memory_capacity
        needed = (
            (self.SL_max / 2 + 1) * self.d_model
            + len(assigned_decoder_blocks)*2*self.SL_max*self.d_model
            + num_param_sum
        ) * self.precision

        return (server_capacity >= needed)

    # ------------------------------------------------------------------------
    # 3) Backtracking to assign servers to each level, enforcing:
    #     - >=1 server per level
    #     - no server in more than one level
    #     - each chosen server passes memory constraints
    #   This returns a list of "partitions" => each is [ [s1, s2..], [...], ...].
    # ------------------------------------------------------------------------
    def _assign_servers_to_levels_with_memory(self, servers, max_L):
        """
        Distribute the given 'servers' among exactly max_L labeled subsets (levels),
        each subset non-empty, skipping servers that end up unused.
        We also check memory constraints for each server in that level's block set.

        Returns a list of feasible partitions:
           [
             [ [srvA, srvB],  [srvC],  [srvD, srvE] ... ],
             [ ... ], ...
           ]
        where each top-level list item is one assignment of servers to levels
        (level 0 => [srvA, srvB], level 1 => [srvC], etc.).
        """

        all_server_ids = list(range(len(servers)))  # server indices
        results = []

        def backtrack(level_index, leftover, current_partition):
            """
            level_index: which level we are assigning
            leftover:    which servers are still not assigned to any level
            current_partition: list of subsets so far, length = max_L
            """
            if level_index == max_L:
                # assigned all levels => store a copy
                results.append([lst[:] for lst in current_partition])
                return

            if not leftover:
                # no servers left for subsequent levels, but we still have levels to fill => fail
                return

            # Step: we want a non-empty subset of leftover for this level,
            # but we only keep servers that pass memory constraints for *this* level
            feasible_servers = []
            for s in leftover:
                if self.check_basic_memory_constraint(s, level_index):
                    feasible_servers.append(s)

            # If no feasible servers remain, we can't fill this level => fail
            if not feasible_servers:
                return

            # We'll try all non-empty subsets of 'feasible_servers' for this level:
            from itertools import chain, combinations
            def all_nonempty_subsets(iterable):
                arr = list(iterable)
                return chain.from_iterable(combinations(arr, r) for r in range(1, len(arr)+1))

            # For each feasible non-empty subset for this level:
            for subset in all_nonempty_subsets(feasible_servers):
                subset = list(subset)
                # Put that subset in level_index
                current_partition[level_index] = subset
                # Remove them from leftover
                new_leftover = [s2 for s2 in leftover if s2 not in subset]
                # Recurse for next level
                backtrack(level_index + 1, new_leftover, current_partition)
                # backtrack
                current_partition[level_index] = []

        current_partition = [[] for _ in range(max_L)]
        backtrack(0, all_server_ids, current_partition)
        return results

    # ------------------------------------------------------------------------
    # 4) Our main "smart brute force" generator
    #    - For each block partition => build Y
    #    - For each servers partition => build X
    #    - Then build Z from X adjacency
    #    - Yields (X, Y, Z)
    # ------------------------------------------------------------------------
    def generate_feasible_solutions(self, servers, num_decoder_blocks, max_L):
        """
        Yields tuples (X, Y, Z) that satisfy:
          - Consecutive blocks per level, and each block in exactly one level (builds Y)
          - Each level has >=1 block
          - Each level has >=1 server
          - Each server is used in <=1 level
          - Memory constraint is satisfied

        The "Z" matrix is built automatically from adjacency of X across consecutive levels.
        """
        # Step 1) All partitions of blocks => consecutive subsets
        block_partitions = self.partition_blocks_into_L_consecutive_subsets(num_decoder_blocks, max_L)

        for block_partition in block_partitions:
            # Build Y_candidate from this block_partition
            # block_partition = [(start0, end0), (start1, end1), ...]
            Y_candidate = [[0]*num_decoder_blocks for _ in range(max_L)]
            for level_index, (start_block, end_block) in enumerate(block_partition):
                for d in range(start_block, end_block+1):
                    Y_candidate[level_index][d] = 1

            # Temporarily assign self.Y so that memory checks reference the correct block set
            old_Y = self.Y
            self.Y = Y_candidate

            # Step 2) All ways to assign servers among the L levels, obeying memory
            server_partitions = self._assign_servers_to_levels_with_memory(servers, max_L)
            # Each element in server_partitions is something like:
            #   [ [0,2], [1], [3,4] ] meaning server-ids 0 and 2 in level0, 1 in level1, 3&4 in level2, etc.

            for sp in server_partitions:
                # Build X
                X_candidate = [[0]*max_L for _ in range(len(servers))]
                for l, server_list_in_level in enumerate(sp):
                    for s_idx in server_list_in_level:
                        X_candidate[s_idx][l] = 1

                # Step 3) Build Z from adjacency across consecutive levels in X
                Z_candidate = [[0 for _ in range(len(servers))] for _ in range(len(servers))]
                for l in range(1, max_L):
                    for s in range(len(servers)):
                        if X_candidate[s][l-1] == 1:
                            for s_prime in range(len(servers)):
                                if X_candidate[s_prime][l] == 1:
                                    Z_candidate[s][s_prime] = 1

                # yield the triple
                yield (X_candidate, Y_candidate, Z_candidate)

            # restore old Y
            self.Y = old_Y

    def check_feasibility_of_solutions(self, solutions):
        valid_solutions = []
        for solution in solutions:
            X, Y, Z = solution
            if Y[0][0] == 1 and Y[self.max_L - 1][self.num_decoder_blocks - 1] == 1:
                valid_solutions.append(solution)
        valid_solutions1 = []
        #one server, one level
        for solution in valid_solutions:
            X, Y, Z = solution
            if all(sum(X[s][l] for l in range(self.max_L)) <= 1 for s in range(len(self.servers))):
                valid_solutions1.append(solution)

        valid_solutions2 = []
        # each level needs at least one decoder block
        for solution in valid_solutions1:
            X, Y, Z = solution
            if all(sum(Y[l][d] for d in range(self.num_decoder_blocks)) >= 1 for l in range(self.max_L)):
                valid_solutions2.append(solution)

        valid_solutions3 = []
            # Each decoder block can be associated with exactly one level
        for solution in valid_solutions2:
            X, Y, Z = solution
            if all(sum(Y[l][d] for l in range(self.max_L)) == 1 for d in range(self.num_decoder_blocks)):
                valid_solutions3.append(solution)

        valid_solutions4 = []
        # Each level needs to be associated with at least one server
        for solution in valid_solutions3:
            X, Y, Z = solution
            if all(sum(X[s][l] for s in range(len(self.servers))) >= 1 for l in range(self.max_L)):
                valid_solutions4.append(solution)

        valid_solutions5 = []
        # Consecutiveness of decoder blocks within a level
        for solution in valid_solutions4:
            X, Y, Z = solution
            consecutive = True
            for l in range(self.max_L):
                decoder_blocks = [d for d in range(self.num_decoder_blocks) if Y[l][d] == 1]
                if decoder_blocks:
                    for i in range(len(decoder_blocks) - 1):
                        if decoder_blocks[i + 1] != decoder_blocks[i] + 1:
                            consecutive = False
                            break
                else:
                    consecutive = False
                    break        
                if not consecutive:
                    break
            if consecutive:
                valid_solutions5.append(solution)

        valid_solutions6 = []
            # Consecutiveness of levels
        for solution in valid_solutions5:
            X, Y, Z = solution
            consecutive_levels = True
            for l in range(1, self.max_L):
                min_d = next((d for d in range(self.num_decoder_blocks) if Y[l][d] == 1), None)
                if min_d is not None:
                   if Y[l-1][min_d-1] != 1:
                       consecutive_levels = False
                else:
                    raise ValueError("min_d is None, which is not supposed to happen.")
                if not consecutive_levels:
                    break
            if consecutive_levels:
                valid_solutions6.append(solution)

        valid_solutions7 = []
        # Check basic memory constraint
        for solution in valid_solutions6:
            X, Y, Z = solution
            memory_constraint_satisfied = True
            for s in range(len(self.servers)):
                for l in range(self.max_L):
                    if X[s][l] == 1 and not self.check_basic_memory_constraint(s, l):
                        memory_constraint_satisfied = False
                        break
                if not memory_constraint_satisfied:
                    break
            if memory_constraint_satisfied:
                valid_solutions7.append(solution)

        print("Number of valid solutions:", len(valid_solutions7))

        return valid_solutions7

    # ------------------------------------------------------------------------
    #  CHANGED FOR SPEED: Evaluate latency in one pass per level
    # ------------------------------------------------------------------------
    def evaluate_latency(self, X, Y, Z):
        """
        Evaluate the objective function based on the given (X, Y, Z) configuration.

        We combine the server-latency and inter-level link-latency retrieval
        into a single loop over levels, so we only call
        `calculate_geomatric_mean_level_latency` once per level.
        """
        total_latency = 0
        # Single pass for levels: server + link latencies
        for l in range(self.max_L):
            avg_server_lat, avg_link_lat = self.calculate_harmonic_mean_level_latency(X, Y, Z, l)
            # Add the server latency for this level
            total_latency += avg_server_lat
            # Add the link latency for connections from level l to level (l+1),
            # except if l == max_L-1 (no next level)
            if l < self.max_L - 1:
                total_latency += avg_link_lat

        # Then add input-layer latency and output-layer latency (as before)
        total_latency += self.calculate_average_latency_first_level_incoming_links(X)
        total_latency += self.calculate_average_latency_last_level_outgoing_links(X)

        return total_latency

    # Other helper methods (identical to your code, unchanged in logic)
    def get_comm_link_latency(self, s, s_prime):
        return self.Z_dict[int(f"{s}{s_prime}")]["object"].get_expected_latency_per_token() * (
            1 / self.get_P_healthy_link(s, s_prime)
        )

    def get_P_healthy_server(self, s, l):
        server_obj = self.X_dict[int(f"{s}{l}")]["object"]
        return server_obj.recovery_rate / (server_obj.recovery_rate + server_obj.failure_rate)

    def get_P_healthy_link(self, s, s_prime):
        link_obj = self.Z_dict[int(f"{s}{s_prime}")]["object"]
        return link_obj.recovery_rate / (link_obj.failure_rate + link_obj.recovery_rate)

    def get_latency_server(self, Y, s, l):
        num_decoder_blocks_in_level = self.determine_num_decoder_blocks_in_level(Y, l)
        server_obj = self.X_dict[int(f"{s}{l}")]["object"]
        average_throughput = server_obj.calculate_average_throughput(num_blocks=num_decoder_blocks_in_level)
        return (1 / average_throughput) * num_decoder_blocks_in_level * (1 / self.get_P_healthy_server(s, l))

    def determine_num_decoder_blocks_in_level(self, Y, l):
        return sum(Y[l][d] for d in range(self.num_decoder_blocks))

    def determine_num_server_in_level(self, X, l):
        return sum(X[s][l] for s in range(len(self.servers)))

    def get_epsilon(self, X, Y, Z, s, l):
        if self.epsilon:
            return self.epsilon
        KV_worst_case = self.KV_worst_case(X, Y, Z, s, l)
        KV_server = self.determine_tok_in_tok_buf_for_server(s, l) * 2 * self.d_model * self.SL_max
        epsilon = (1 - min(1, KV_server / KV_worst_case))
        return epsilon

    def get_delta(self, X, Y, Z, s, l):
        server_obj = self.X_dict[int(f"{s}{l}")]["object"]
        num_decoder_blocks_in_level = self.determine_num_decoder_blocks_in_level(Y, l)
        average_throughput = server_obj.calculate_average_throughput(num_blocks=num_decoder_blocks_in_level)
        # Avoid dividing by zero
        denom = self.calculate_average_dwell_time_per_token_in_iteration(X, Y, Z) * average_throughput
        if denom == 0:
            return 1
        return min(self.determine_tok_in_tok_buf_for_server(s, l) / int(denom), 1)

    def KV_worst_case(self, X, Y, Z, s, l):
        server_obj = self.X_dict[int(f"{s}{l}")]["object"]
        num_decoder_blocks_in_level = self.determine_num_decoder_blocks_in_level(Y, l)
        average_throughput = server_obj.calculate_average_throughput(num_blocks=num_decoder_blocks_in_level)
        dwell = self.calculate_average_dwell_time_per_token_in_iteration(X, Y, Z)
        worst_case_KV = int(dwell * average_throughput) * 2 * self.d_model * self.SL_max
        return worst_case_KV

    def determine_tok_in_tok_buf_for_server(self, s, l):
        assigned_decoder_blocks = [d for d in range(self.num_decoder_blocks) if self.Y[l][d] == 1]
        num_param_sum = sum(self.exemplary_decoder_blocks[d].num_param for d in assigned_decoder_blocks)
        one_KV_cache_server = 2 * self.SL_max * self.d_model
        available_memory = self.X_dict[int(f"{s}{l}")]["object"].cache_memory_capacity
        # This formula is from your code, unchanged
        tok = (
            ((available_memory / self.precision) - num_param_sum - (self.SL_max / 2) * self.d_model)
            / ((len(assigned_decoder_blocks) * one_KV_cache_server + 1) * self.d_model)
        )
        return int(tok)

    def calculate_average_dwell_time_per_token_in_iteration(self, X, Y, Z):
        dwell_sys = 0
        for l in range(self.max_L):
            server_lat = 0
            num_servers_in_level = self.determine_num_server_in_level(X, l)
            if num_servers_in_level == 0:
                continue
            lat_sum = 0
            count_srv = 0
            for s in range(len(self.servers)):
                if X[s][l] == 1:
                    num_blocks = self.determine_num_decoder_blocks_in_level(Y, l)
                    server_obj = self.X_dict[int(f"{s}{l}")]["object"]
                    throughput = server_obj.calculate_average_throughput(num_blocks=num_blocks)
                    if num_blocks > 0 and throughput > 0:
                        lat_sum += (1 / throughput) * num_blocks
                        count_srv += 1
            # Average over servers in this level
            if count_srv > 0:
                server_lat = lat_sum / count_srv
            dwell_sys += server_lat

            # Now average latency of staged communication links
            for ll in range(self.max_L - 1):
                total_latency = 0
                count_links = 0
                for s in range(len(self.servers)):
                    for s_prime in range(len(self.servers)):
                        if Z[s][s_prime] == 1:
                            link_obj = self.Z_dict[int(f"{s}{s_prime}")]["object"]
                            total_latency += link_obj.get_expected_latency_per_token()
                            count_links += 1
                average_latency = total_latency / count_links if count_links > 0 else 0
                dwell_sys += average_latency

        # Add 2 * averaged link latency across all server pairs
        total_latency = 0
        count_links = 0
        for s in range(len(self.servers)):
            for s_prime in range(len(self.servers)):
                if s != s_prime:
                    link_obj = self.Z_dict[int(f"{s}{s_prime}")]["object"]
                    total_latency += link_obj.get_expected_latency_per_token()
                    count_links += 1
        if count_links > 0:
            dwell_sys += 2 * (total_latency / count_links)
        return dwell_sys

    def calculate_harmonic_mean_level_latency(self, X, Y, Z, l):
        # This function returns (average_latency_server, average_latency_comm_link)
        latencies_servers = []
        for s in range(len(self.servers)):
            if X[s][l] == 1:
                epsilon = self.get_epsilon(X, Y, Z, s, l)
                delta = self.get_delta(X, Y, Z, s, l)
                base_lat = self.get_latency_server(Y, s, l)
                # Weighted formula from your code
                latency = base_lat * ((1 - epsilon) * (1 / delta) + epsilon * 10)
                latencies_servers.append(latency)

        # Server portion: "harmonic concurrency" example from your code
        # total_latency_server = 1 / sum(1 / lat for lat in latencies_servers)
        # but you do an unusual approach: we’ll replicate exactly your code
        if latencies_servers:
            total_latency_server = 1 / sum((1 / lat) for lat in latencies_servers)
        else:
            total_latency_server = 0
            raise ValueError("Unfeasible solution: Levels are not consecutive.")


                    # Communication links from level l to l+1
        
        total_latency_comm_link = 0
        if l < self.max_L - 1:
            count = 0
            for s in range(len(self.servers)):
                if X[s][l] == 1:
                    latencies_comm_links = []
                    for s_prime in range(len(self.servers)):
                        if Z[s][s_prime] == 1:
                            latencies_comm_links.append(self.get_comm_link_latency(s, s_prime))
                            
                    total_latency_comm_link += 1 / sum(1 / lat for lat in latencies_comm_links)      

                    count+=1
                

        if total_latency_comm_link:
            # You do a partial "harmonic mean" approach with a final /lensel s:
            total_latency_comm_link = total_latency_comm_link /count  
        else:
            total_latency_comm_link = 0

        # We return them separately to let evaluate_latency decide how to add them
        return (total_latency_server, total_latency_comm_link)

    def calculate_average_latency_last_level_outgoing_links(self, X):
        total_latency = 0
        count_links = 0
        for s in range(len(self.servers)):
            if X[s][self.max_L - 1] == 1:
                comm_link_list = []
                for s_prime in range(len(self.servers)):
                    if s != s_prime:
                        comm_link_list.append(self.get_comm_link_latency(s, s_prime))
                if comm_link_list:
                    # Weighted combination
                    total_latency += 1 / sum((1 / lat) for lat in comm_link_list)
                count_links += 1
        average_latency = total_latency / count_links if count_links > 0 else 0
        return average_latency

    def calculate_average_latency_first_level_incoming_links(self, X):
        total_latency = 0
        count_links = 0
        for s_prime in range(len(self.servers)):
            comm_link_list = []
            for s in range(len(self.servers)):
                if s != s_prime and X[s][0] == 1:
                    comm_link_list.append(self.get_comm_link_latency(s_prime, s))
            if comm_link_list:
                total_latency += 1 / sum((1 / lat) for lat in comm_link_list)
            count_links += 1
        average_latency = total_latency / count_links if count_links > 0 else 0
        return average_latency

    def solve(self):
        """
        Solve the optimization problem using the 'smart brute force'.
        We generate feasible solutions, do final checks, then evaluate latencies.
        """
        print("Generating (X,Y,Z) solutions with pre-pruning...")
        all_solutions = []
        for (X, Y, Z) in self.generate_feasible_solutions(self.servers, self.num_decoder_blocks, self.max_L):
            all_solutions.append((X, Y, Z))

        print("Number of 'pre-pruned' solutions:", len(all_solutions))
        print("Checking feasibility of solutions...")
        valid_solutions = self.check_feasibility_of_solutions(all_solutions)
        print("Evaluating latencies of feasible solutions...")
        start_time = time.time()
        i = 0
        for X, Y, Z in valid_solutions:
            i += 1
            if i % 1000 == 0:
                progress = (i / len(valid_solutions)) * 100
                print(f"Evaluating solution {i}/{len(valid_solutions)} ({progress:.2f}%)")
                if i == 5000:
                    elapsed_time = time.time() - start_time
                    estimated_total_time = (elapsed_time / 5000) * len(valid_solutions)
                    estimated_remaining_time = estimated_total_time - elapsed_time
                    estimated_total_time_min = estimated_total_time / 60
                    estimated_remaining_time_min = estimated_remaining_time / 60
                    print(f"Estimated total time: {estimated_total_time:.2f} seconds ({estimated_total_time_min:.2f} minutes)")
                    print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds ({estimated_remaining_time_min:.2f} minutes)")
            # Evaluate the objective (latency)
            latency = self.evaluate_latency(X, Y, Z)
            if latency < self.best_latency:
                self.best_latency = latency
                self.best_solution = (X, Y, Z)
        print("Best latency:", self.best_latency)
        
        return self.best_solution, self.best_latency

    def make_info_dict(self, X, Y, Z):
        swarm_info = {}
        for l in range(self.max_L):
            swarm_id = l + 1
            server_ids = [self.X_dict[int(f"{s}{l}")]["server_id"] for s in range(len(self.servers)) if X[s][l] == 1]
            num_decoder_blocks = sum(Y[l][d] for d in range(self.num_decoder_blocks))
            decoder_block_ids = [d for d in range(self.num_decoder_blocks) if Y[l][d] == 1]
            comm_link_ids = [
                self.Z_dict[int(f"{s}{s_prime}")]["comm_link_id"]
                for s in range(len(self.servers)) if X[s][l] == 1
                for s_prime in range(len(self.servers)) if Z[s][s_prime] == 1
            ]
            swarm_info[swarm_id] = {
                "num_decoder_blocks": num_decoder_blocks,
                "decoder_block_ids": decoder_block_ids,
                "server_ids": server_ids,
                "comm_link_ids": comm_link_ids
            }
        return swarm_info
    
    def translate(self, X, Y, Z):
        """
        Translate the X, Y, Z matrices to the corresponding server and communication link IDs.
        """
        translated_X = [
            [self.X_dict[int(f"{s}{l}")]["server_id"] if X[s][l] == 1 else 0 for l in range(self.max_L)]
            for s in range(len(self.servers))
        ]
        translated_Y = Y
        translated_Z = [
            [
                self.Z_dict[int(f"{s}{s_prime}")]["comm_link_id"] if Z[s][s_prime] == 1 else 0
                for s_prime in range(len(self.servers))
            ]
            for s in range(len(self.servers))
        ]
        result_str = ""
        for l in range(self.max_L):
            servers_in_level = [
                self.X_dict[int(f"{s}{l}")]["server_id"]
                for s in range(len(self.servers))
                if X[s][l] == 1
            ]
            result_str += f"level {l}: servers {', '.join(map(str, servers_in_level))}; "
            if l < self.max_L - 1:
                comm_links = [
                    self.Z_dict[int(f'{s}{s_prime}')]["comm_link_id"]
                    for s in range(len(self.servers))
                    for s_prime in range(len(self.servers))
                    if Z[s][s_prime] == 1
                ]
                result_str += (
                    f"comm_links from level {l} to level {l + 1}: "
                    f"{', '.join(map(str, comm_links))}; "
                )
        print(result_str)
        return translated_X, translated_Y, translated_Z



    def check_cacheKV_constraint(self, cacheKV_worst, cacheKV, cacheKV_server_mem_limit, G_KVcache):
        return all(cacheKV_worst[s] + cacheKV[s] + cacheKV_server_mem_limit[s] <= G_KVcache[s] for s in cacheKV)

    def check_min_servers_per_layer(self, x, G_min_server, L):
        return all(sum(x[s, l] for s in x if (s, l) in x) >= G_min_server[l] for l in range(1, L+1))

    def check_one_layer_per_server(self, x, S):
        return all(sum(x[s, l] for l in range(1, L+1) if (s, l) in x) <= 1 for s in S)

    def check_min_decoder_per_layer(self, y, L, D_set):
        return all(sum(y[l, d] for d in D_set if (l, d) in y) >= 1 for l in range(1, L+1))

    def check_one_layer_per_decoder(self, y, D_set, L):
        return all(sum(y[l, d] for l in range(1, L+1) if (l, d) in y) == 1 for d in D_set)

    def check_z_x_constraints(self, z, x, S, L):
        return all(
            z[s, s_prime] <= x.get((s, l - 1), 0) and
            z[s, s_prime] <= x.get((s_prime, l), 0) and
            z[s, s_prime] >= x.get((s, l - 1), 0) + x.get((s_prime, l), 0) - 1 and
            z[s, s_prime] in {0, 1}
            for s in S for s_prime in S for l in range(1, L + 1) if (s, s_prime) in z
        )

    def check_one_block_per_level(self, y, L, D_set):
        return all(sum(y.get((l, d), 0) for d in D_set) >= 1 for l in range(1, L+1))

    def check_define_alpha_l(self, y, alpha, L, D_set, M):
        return all(
            (sum(y.get((l, d), 0) for d in D_set) - 1 >= alpha.get(l, 0) and
            sum(y.get((l, d), 0) for d in D_set) - 1 <= M * alpha.get(l, 0))
            for l in range(1, L+1)
        )

    def check_contiguity(self, y, alpha, L, D_set, M):
        return all(
            y.get((l, d), 0) <= y.get((l, d - 1), 0) + y.get((l, d + 1), 0) + M * (1 - alpha.get(l, 0))
            for l in range(1, L+1) for d in range(2, len(D_set))
        ) and all(
            y.get((l, 1), 0) <= y.get((l, 2), 0) + M * (1 - alpha.get(l, 0)) and
            y.get((l, max(D_set)), 0) <= y.get((l, max(D_set) - 1), 0) + M * (1 - alpha.get(l, 0))
            for l in range(1, L+1)
        )

    def check_decoder_ordering(self, y, L, D_set):
        return all(
            y.get((l, d), 0) + y.get((l_prime, d_prime), 0) <= 1
            for l in range(1, L+1) for l_prime in range(1, L+1)
            for d in D_set for d_prime in D_set if l > l_prime and d < d_prime
        )
