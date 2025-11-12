import copy
import math
import random
import numpy as np
from collections import defaultdict
from typing import Optional
from graphviz import Digraph, Source

class MCTSNode:
    def __init__(self, parent=None, action=None, state=None, done=False):
        """
        parent: parent node
        action: action that led from parent -> this node
        state: environment observation or full env state (depending on your env)
        done: whether this state is terminal
        """
        self.parent = parent
        self.action = action
        self.state = state
        self.done = done

        # MCTS statistics
        self.children = {}  # action_int -> MCTSNode
        self.visit_count = 0
        self.cum_reward_vec = None
        self.utility_vec = None
        self.untried_actions = None

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

class MCTS:
    def __init__(self, env, n_objectives=None, rollout_depth=10, n_simulations=5_000,
                 exploration_const=math.sqrt(2), gamma=0.95, mode=None, rng=None):
        self.env = env
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.C = exploration_const
        self.gamma = gamma
        self.mode = env.mode
        self.rng = np.random.default_rng() if rng is None else rng
        self.n_objectives = n_objectives or 2  # default to search + track
        # Precompute action space for convenience if env provides
        if hasattr(env, "action_space"):
            self.action_space_size = env.action_space.n
        else:
            self.action_space_size = None

    def _select(self, node):
        """Select child node based on multi-objective UCB with Pareto dominance."""
        if not node.children:
            return node  # no children yet

        children_list = list(node.children.values())
        n_parent = max(1, node.visit_count)
        utilities = np.zeros(len(children_list))
        ucb = np.zeros(len(children_list))

        # --- Construct [searchR, trackR] vectors ---
        total_rtrack = []
        for child in children_list:
            avg_util = child.utility_vec if child.utility_vec is not None else np.zeros(self.n_objectives)
            searchR = avg_util[0]
            trackR = np.sum(avg_util[1:])
            total_rtrack.append([searchR, trackR])

        # --- Pareto-based dominance filtering ---
        for i, child in enumerate(children_list):
            removed = total_rtrack.pop(i)
            dominating_vecs = self._get_dominating_vecs(total_rtrack, removed, maximize=[True, True])
            pareto_penalty = -len(dominating_vecs)
            utilities[i] = pareto_penalty
            total_rtrack.insert(i, removed)

        # --- Compute UCB ---
        for i, child in enumerate(children_list):
            n_child = max(1, child.visit_count)
            base_utility = np.sum(child.utility_vec) if child.utility_vec is not None else 0.0
            ucb[i] = base_utility + utilities[i] + self.C * math.sqrt(math.log(n_parent) / n_child)

        # --- Select max UCB (break ties randomly) ---
        max_ucb = np.max(ucb)
        best_indices = [i for i, val in enumerate(ucb) if val == max_ucb]
        return children_list[random.choice(best_indices)]

    def _expand(self, node, env):
        """Randomly sample a macro/micro action based on mode and add a child node."""
        # 1. Sample macro action
        if self.mode == "search":
            macro_action = 0
        elif self.mode == "track":
            macro_action = 1
        else:  # both
            macro_action = np.random.choice([0, 1])

        # 2. Sample micro action
        if macro_action == 0:  # SEARCH
            micro_search = np.random.choice(len(env.grid_coords))
            micro_track = None
        else:  # TRACK
            known_targets = np.where(env.known_mask)[0]
            if len(known_targets) == 0:
                # fallback to search
                macro_action = 0
                micro_search = np.random.choice(len(env.grid_coords))
                micro_track = None
            else:
                micro_track = np.random.choice(known_targets)
                micro_search = None

        # 3. Encode and step action
        action = env.encode_action(macro_action, micro_search, micro_track)
        next_obs, rewards, done, truncated, info = env.step(action)
        if info.get("invalid_action", False):
            return None, None

        # 4. Create child node
        child_node = MCTSNode(parent=node, action=action, state=next_obs, done=done)
        child_node.cum_reward_vec = np.zeros(self.n_objectives)
        child_node.utility_vec = np.zeros(self.n_objectives)
        node.children[action] = child_node

        return child_node, action

    def _rollout(self, env, max_depth=None):
        max_depth = max_depth or self.rollout_depth
        total_reward = np.zeros(self.n_objectives)
        
        for _ in range(max_depth):
            # Sample macro action
            if self.mode == "search":
                macro_action = 0
            elif self.mode == "track":
                macro_action = 1
            else:
                macro_action = np.random.choice([0, 1])
            
            # Sample micro action
            if macro_action == 0:  # SEARCH
                micro_search = np.random.choice(len(env.grid_coords))
                micro_track = None
            else:  # TRACK
                known_targets = np.where(env.known_mask)[0]
                if len(known_targets) == 0:
                    macro_action = 0
                    micro_search = np.random.choice(len(env.grid_coords))
                    micro_track = None
                else:
                    micro_track = np.random.choice(known_targets)
                    micro_search = None
            
            # Encode and execute
            action = env.encode_action(macro_action, micro_search, micro_track)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Build proper reward vector [searchR, trackR]
            reward_vec = np.zeros(self.n_objectives)
            if info["macro"] == 0:
                reward_vec[0] = reward  # search reward
            elif info["macro"] == 1:
                reward_vec[1] = reward  # track reward
            
            total_reward += reward_vec

            if done or truncated:
                break
        
        return total_reward

    def _backprop(self, path, reward_vec):
        """Backpropagate discounted reward vector along path."""
        discounted_reward = np.array(reward_vec, dtype=float)
        for node in reversed(path):
            node.visit_count += 1
            if node.utility_vec is None:
                node.utility_vec = discounted_reward.copy()
            else:
                node.utility_vec += (discounted_reward - node.utility_vec) / node.visit_count
            if node.cum_reward_vec is None:
                node.cum_reward_vec = discounted_reward.copy()
            else:
                node.cum_reward_vec += discounted_reward
            discounted_reward *= self.gamma

    def _get_dominating_vecs(self, all_vectors, to_compare, order):
        dominating = []
        for vec in all_vectors:
            dominates = True
            for v, t, maximize in zip(vec, to_compare, order):
                if maximize:
                    if v < t:  # worse in a maximization objective
                        dominates = False
                        break
                else:
                    if v > t:  # worse in a minimization objective
                        dominates = False
                        break
            if dominates and not np.allclose(vec, to_compare):
                dominating.append(vec)
        return dominating

    def choose(self, root_env=None):
        """Run MCTS and return best action from root node."""
        root_env = root_env or self.env
        root = MCTSNode(parent=None, action=None, state=root_env)
        root.cum_reward_vec = np.zeros(self.n_objectives)
        root.utility_vec = np.zeros(self.n_objectives)

        for _ in range(self.n_simulations):
            env_copy = copy.deepcopy(root_env)
            node = root
            path = [node]

            # Selection
            while node.is_fully_expanded() and node.children:
                node = self._select(node)
                # Step environment
                next_obs, rewards, done, truncated, info = env_copy.step(node.action)
                path.append(node)
                if done or truncated:
                    break

            # Expansion
            if not node.is_fully_expanded():
                child, _ = self._expand(node, env_copy)
                if child is not None:
                    node = child
                    path.append(node)

            # Rollout
            reward_vec = self._rollout(env_copy, self.rollout_depth)

            # Backprop
            self._backprop(path, reward_vec)

        # Pick action with highest visit count
        best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        return best_action
    
    def reset_tree(self, obs):
        """Set up new root node for a fresh planning phase."""
        self.root = MCTSNode(state=obs)

    def re_root(self, action):
        """Re-root the tree after taking `action`."""
        if action in self.root.children:
            new_root = self.root.children[action]
            new_root.parent = None
            return new_root
        else:
            # start a new tree if the branch wasn’t expanded
            return MCTSNode(state=None)

    def visualize_mcts_tree(self, root, max_depth=3, highlight_best=True):
        """
        Visualize a multi-objective hierarchical MCTS tree (purely in-memory, no system Graphviz needed).

        Parameters
        ----------
        root : MCTSNode
            Root node of the tree
        max_depth : int
            Maximum depth to visualize
        highlight_best : bool
            If True, color nodes with highest cumulative reward
        """
        dot = Digraph()

        # --- 1. Compute best cumulative reward if highlighting ---
        best_reward = -np.inf
        if highlight_best:
            def get_best_reward(node):
                nonlocal best_reward
                if node.cum_reward_vec is not None:
                    reward_sum = np.sum(node.cum_reward_vec)
                    best_reward = max(best_reward, reward_sum)
                for child in node.children.values():
                    get_best_reward(child)
            get_best_reward(root)

        # --- 2. Recursive function to add nodes ---
        def add_node(node, depth=0):
            if depth > max_depth:
                return
            
            node_id = str(id(node))
            reward_vec = node.cum_reward_vec if node.cum_reward_vec is not None else np.zeros(2)
            visits = node.visit_count
            label = f"Visits: {visits}\nReward: [{reward_vec[0]:.2f}, {reward_vec[1]:.2f}]"

            # Highlight node with best reward
            color = "black"
            if highlight_best and np.sum(reward_vec) == best_reward:
                color = "red"

            dot.node(node_id, label=label, color=color)

            # Add edges to children
            for action, child in node.children.items():
                child_id = str(id(child))
                # Decode hierarchical action for display
                if isinstance(action, tuple) and len(action) == 3:
                    macro, micro_search, micro_track = action
                    if macro == 0:
                        action_label = f"S:{micro_search}"
                    else:
                        action_label = f"T:{micro_track}"
                else:
                    action_label = str(action)
                dot.edge(node_id, child_id, label=action_label)
                add_node(child, depth + 1)

        add_node(root)

        # --- 3. Return a Source object for in-memory display ---
        return Source(dot.source)