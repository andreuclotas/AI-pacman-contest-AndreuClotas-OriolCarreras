# my_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random
import util
from capture_agents import CaptureAgent
from game import Directions
from collections import deque
from util import PriorityQueue

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    
    '''
    Two different agents to form the team, an offensive and a defensive agent
    '''
    return [eval(first)(first_index), eval(second)(second_index)]


############################################################
# 1. BASE AGENT CLASS
############################################################

class ReflexCaptureAgent(CaptureAgent):
    """
    Base class for both our agents. It handles all the common logic like:
    - analyzing the map layout
    - tracking enemy positions (even when they are invisible)
    - choosing the best action based on features and weights
    """
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        
        # Map Analysis, we precompute info at the start so we don't have to do it every turn
        self.start_pos = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        self.width = self.walls.width
        self.height = self.walls.height
        
        # identify midline boundary points
        if self.red:
            self.mid_x = int((self.width / 2) - 1)
        else:
            self.mid_x = int(self.width / 2)
            
        # Store all accessible boundary points (points that are not walls)    
        self.boundary_points = []
        for y in range(self.height):
            if not self.walls[self.mid_x][y]:
                self.boundary_points.append((self.mid_x, y))
        
        # Dictionary to remember last known enemy positions
        self.enemy_memory = {} 
        # Keep trach of current food and previous defending food
        self.current_food_list = []
        self.prev_defending_food = None
        # Game state snapshot
        self.game_state = None 
        # Recent positions to prevent getting stuck in never-ending loops
        self.recent_positions = deque(maxlen=10)

    def choose_action(self, game_state):
        """
        Main decision loop called every turn.
        1. Update our perception of the world (where are enemies? did they eat food?)
        2. Decide on a high-level target (e.g., specific food, home, invader)
        3. Evaluate legal moves to see which one gets us closer to that target safely
        """

        # update perception
        self.update_perception(game_state)

        # pick a target
        target = self.select_target()
        
        # Store position to avoid getting stuck
        my_pos = game_state.get_agent_position(self.index)
        if my_pos:
            self.recent_positions.append((int(my_pos[0]), int(my_pos[1])))

        # Get legal actions
        actions = game_state.get_legal_actions(self.index)

        # Avoid stopping if there are other possible actions
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)
            
        best_action = None
        best_val = -float('inf')
        
        # Check all actions
        for action in actions:

            # get successor
            successor = self.get_successor(game_state, action)

            # get features and weights
            features = self.get_features(successor, action, target)
            weights = self.get_weights(successor, action, target)
            val = features * weights
            
            # keep the move with the best value
            if val > best_val:
                best_val = val
                best_action = action
                
        return best_action

    def update_perception(self, game_state):
        """
        Updates internal state, tracks enemies, and snapshots food
        """
        self.game_state = game_state
        self.my_pos = game_state.get_agent_position(self.index)
        self.my_state = game_state.get_agent_state(self.index)
        self.current_food_list = self.get_food(game_state).as_list()
        
        # If food dissapears on our side but we can't see the enemy,
        # update memory for the closest unknown enemy
        current_defending = self.get_food_you_are_defending(game_state).as_list()
        if self.prev_defending_food:

            # Check for eaten food
            eaten_food = set(self.prev_defending_food) - set(current_defending)
            if eaten_food:
                eaten_pos = list(eaten_food)[0]
                # Assign to closest unseen opponent
                opponents = self.get_opponents(game_state)
                for opp_idx in opponents:
                    if game_state.get_agent_position(opp_idx) is None:
                        self.enemy_memory[opp_idx] = eaten_pos
                        break
        self.prev_defending_food = current_defending

        # Update enemy positions and classify as invaders or defenders
        opponents = self.get_opponents(game_state)
        self.invaders = []
        self.defenders = []
        
        for opp_idx in opponents:
            state = game_state.get_agent_state(opp_idx)
            pos = state.get_position()
            
            if pos:
                # If we see them, update memory
                pos = (int(pos[0]), int(pos[1]))
                self.enemy_memory[opp_idx] = pos
            
            # Use last known position if not visible
            final_pos = pos if pos else self.enemy_memory.get(opp_idx)
            
            if final_pos:
                if state.is_pacman:
                    self.invaders.append((opp_idx, state, final_pos))
                else:
                    self.defenders.append((opp_idx, state, final_pos))

    def is_safe(self, pos, safe_dist=2):
        """
        Safety check: returns False if our position is dangerously close to an active (enemy) ghost
        We ignore ghosts that are scared or far away
        """
        for idx, state, enemy_pos in self.defenders:
            # It's only dangerous if they are not pacman and their scared timer is low
            if not state.is_pacman and state.scared_timer <= 5:
                if self.get_maze_distance(pos, enemy_pos) <= safe_dist:
                    return False
        return True

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        return successor

    def get_closest_boundary(self, pos):

        # Finds closest boundary point to given position
        return min(self.boundary_points, key=lambda p: self.get_maze_distance(pos, p))
    
    def get_safe_maze_distance(self, start_pos, target_pos):
        """
        Modified get_maze_distance() that treats dangerous ghosts as if they were walls
        This prevents paths that go too close to active ghosts

        Returns a large distance (500) if no safe path exists,
        this avoids overriding other penalties in feature calculations
        
        """
        
        
        # Identify unsafe zones (Ghost position + 1 tile radius)
        unsafe = set()
        for idx, state, pos in self.defenders:
            if not state.is_pacman and state.scared_timer <= 5:
                unsafe.add(pos)
                x, y = pos
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    unsafe.add((int(x+dx), int(y+dy)))
                    
        queue = PriorityQueue()
        queue.push((start_pos, 0), 0)
        visited = set()
        
        while not queue.is_empty():
            curr, dist = queue.pop()
            
            if curr == target_pos:
                return dist
            
            if curr in visited: 
                continue
            visited.add(curr)
            
            # don't search forever
            if dist > 100: 
                return 500

            x, y = curr
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_pos = (int(x+dx), int(y+dy))
                # only consider safe positions
                if not self.walls[next_pos[0]][next_pos[1]] and next_pos not in unsafe:
                    priority = dist + 1 + util.manhattan_distance(next_pos, target_pos)
                    queue.push((next_pos, dist + 1), priority)
        
        return 500 

    def select_target(self): 
        return self.start_pos
    
    def get_features(self, game_state, action, target): 
        return util.Counter()
    
    def get_weights(self, game_state, action, target): 
        return util.Counter()


############################################################
# 2 OFFENSIVE AGENT
############################################################

class OffensiveAgent(ReflexCaptureAgent):
    """
    The Attacker
    Strategy:
    1. Eat food
    2. If chased, run to a capsule or back home
    3. If carrying too much food, return home to score
    """
    def select_target(self):
        food_list = self.current_food_list
        capsules = self.get_capsules(self.game_state)
        carrying = self.my_state.num_carrying
        time_left = self.game_state.data.timeleft
        score = self.get_score(self.game_state)
        
        # If winning significantly, play safe
        if score > 7:
            return self.get_closest_boundary(self.my_pos)

        # Ghost Hunting
        # If there are scared ghosts, chase the closest one
        scared_ghosts = [p for p in self.defenders if p[1].scared_timer > 5]
        if scared_ghosts:
            closest_ghost = min(scared_ghosts, key=lambda g: self.get_maze_distance(self.my_pos, g[2]))
            return closest_ghost[2]

        # Survival
        # If in danger, run to capsule or home
        if not self.is_safe(self.my_pos, safe_dist=4):
            if capsules:
                closest_cap = min(capsules, key=lambda c: self.get_maze_distance(self.my_pos, c))
                # Only run to capsule if we can reach it safely
                if self.is_safe(closest_cap, safe_dist=1):
                    return closest_cap
            # Otherwise run home
            best_bp = min(self.boundary_points, key=lambda p: self.get_safe_maze_distance(self.my_pos, p))
            return best_bp
            
        # Score
        # If carrying enough food, or few food left, or time is running out, return home
        if carrying >= 5 or len(food_list) <= 2 or time_left < 60:
            return min(self.boundary_points, key=lambda p: self.get_safe_maze_distance(self.my_pos, p))
            
        # Attack
        # Go for the best food based on distance and risk
        if food_list:
            return max(food_list, key=lambda f: self.score_food(f, self.my_pos))
            
        return self.start_pos

    def score_food(self, food_pos, my_pos):
        """
        Heuristic to pick the 'best' food
        Prefer food that is close to us but far from ghosts
        """
        dist = self.get_maze_distance(my_pos, food_pos)
        risk = 0
        active_ghosts = [p[2] for p in self.defenders 
                         if not p[1].is_pacman and p[1].scared_timer <= 5]
        if active_ghosts:
            min_ghost_dist = min([self.get_maze_distance(food_pos, g_pos) for g_pos in active_ghosts])
            # Penalize food that is too close to ghosts
            if min_ghost_dist < 6: 
                risk = -200.0 / (min_ghost_dist + 0.1)
        return -dist + risk

    # Features
    def get_features(self, successor, action, target):
        features = util.Counter()
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        my_pos_int = (int(my_pos[0]), int(my_pos[1]))
        
        # Distance to target, use safe distance if carrying food or in danger
        if self.my_state.num_carrying > 0 or features['danger'] == 1:
             dist = self.get_safe_maze_distance(my_pos_int, target)
        else:
             dist = self.get_maze_distance(my_pos_int, target)
        
        features['dist_to_target'] = dist
        
        # Trapped
        # If no safe path to target exists, consider ourselves trapped
        if dist >= 500: 
            features['trapped'] = 1

        # Eating Food / Capsules
        if my_pos_int in self.current_food_list:
            features['eats_food'] = 1
            
        current_capsules = self.get_capsules(self.game_state)
        if my_pos_int in current_capsules:
            features['eats_capsule'] = 1
        
        # Safety Features
        if not self.is_safe(my_pos_int, 1): 
            features['death'] = 1
        elif not self.is_safe(my_pos_int, 2): 
            features['danger'] = 1
        
        # If only one legal action, we're in a dead end
        if my_state.is_pacman:
             actions = successor.get_legal_actions(self.index)
             if len(actions) <= 1: features['dead_end'] = 1
        
        # Check if we've visited this position recently
        if my_pos_int in self.recent_positions:
            features['visited_recently'] = 1

        # Penalize reversing direction
        current_dir = self.game_state.get_agent_state(self.index).configuration.direction
        if action == Directions.REVERSE[current_dir]:
            features['reverse'] = 1

        return features

    # Weights
    def get_weights(self, game_state, action, target):
        return {
            'dist_to_target': -10, 
            'eats_food': 100, 
            'eats_capsule': 200, 
            'death': -50000,    # Massive penalty to override other factors
            'danger': -2000,    # Significant penalty for being in danger
            'trapped': -5000,   # High penalty for being trapped
            'dead_end': -50,
            'visited_recently': -40, # discourage loops
            'reverse': -5 
        }


############################################################
# 3 DEFENSIVE AGENT
############################################################

class DefensiveAgent(ReflexCaptureAgent):
    """
        We divide the map into Top, Middle, and Bottom zones
        We find the center of our food in each zone and create a patrol route
        This way we cover all areas of our territory effectively
    """
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.patrol_index = 0
        self.patrol_points = self.calculate_patrol_points(game_state)

    def calculate_patrol_points(self, game_state):
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if not food_list:
            return self.boundary_points
            
        height = self.height
        zones = {0: [], 1: [], 2: []}
        
        for f in food_list:
            if f[1] < height // 3: zones[2].append(f) # Bottom zone
            elif f[1] < 2 * height // 3: zones[1].append(f) # Middle zone
            else: zones[0].append(f) # Top zone
            
        patrol_points = []
        for i in range(3):
            if zones[i]:
                # Find center of food in this zone
                avg_x = sum(f[0] for f in zones[i]) / len(zones[i])
                avg_y = sum(f[1] for f in zones[i]) / len(zones[i])
                
                # Find the food closest to this center,
                # so we don't go to a wall
                central_food = min(zones[i], key=lambda f: (f[0]-avg_x)**2 + (f[1]-avg_y)**2)
                
                # From that food, find the closest boundary point
                best_bp = min(self.boundary_points, 
                              key=lambda p: self.get_maze_distance(p, central_food))
                patrol_points.append(best_bp)
                
        # Remove duplicates and sort by y-coordinate for consistent patrol order
        unique = list(set(patrol_points))
        unique.sort(key=lambda p: p[1]) 
        return unique

    def select_target(self):
        # Kill invaders if any are present
        if self.invaders:
            targets = [i for i in self.invaders if i[1].is_pacman]
            if targets:
                return min(targets, key=lambda t: self.get_maze_distance(self.my_pos, t[2]))[2]

        # Investigate last known enemy positions on our side
        memory_targets = []
        for idx, pos in self.enemy_memory.items():
            is_our_side = False
            if self.red and pos[0] < self.mid_x: is_our_side = True
            if not self.red and pos[0] > self.mid_x: is_our_side = True
            if is_our_side: memory_targets.append(pos)
                
        if memory_targets:
             return min(memory_targets, key=lambda p: self.get_maze_distance(self.my_pos, p))

        # Patrol
        # If no last known positions, follow patrol route
        if not self.patrol_points:
            self.patrol_points = self.calculate_patrol_points(self.game_state)
            
        if self.patrol_points:
            current_patrol = self.patrol_points[self.patrol_index]
            # Move to next patrol point if we've reached the current one
            if self.get_maze_distance(self.my_pos, current_patrol) <= 2:
                self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
                return self.patrol_points[self.patrol_index]
            return current_patrol
            
        return self.boundary_points[0]

    # Features
    def get_features(self, successor, action, target):
        features = util.Counter()
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Distance to target, no need for safe distance here
        features['dist_to_target'] = self.get_maze_distance(my_pos, target)
        
        # Killing invaders
        if self.invaders:
            for i in self.invaders:
                if my_pos == i[2] and my_state.scared_timer == 0:
                    features['kill_invader'] = 1

        # Stay on defense
        if my_state.is_pacman: features['on_defense'] = 0
        else: features['on_defense'] = 1

        # Penalize reversing direction    
        current_dir = self.game_state.get_agent_state(self.index).configuration.direction
        if action == Directions.REVERSE[current_dir]:
            features['reverse'] = 1
        
        return features

    # Weights
    def get_weights(self, game_state, action, target):
        return {
            'dist_to_target': -1, 
            'kill_invader': 1000, # big reward for killing invaders
            'on_defense': 500, # strongly prefer being on defense
            'reverse': -2
        }