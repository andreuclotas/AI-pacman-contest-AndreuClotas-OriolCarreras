# baseline_team.py
import random
import util
import time

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class BaseCaptureAgent(CaptureAgent):
    
    # Shared utilities for both agents.
    

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        
        # Team color
        if self.index in game_state.get_red_team_indices():
            self.team = 'red'
            self.enemy_team = 'blue'
        else:
            self.team = 'blue' 
            self.enemy_team = 'red'
            
        # Precompute boundary
        self.boundary = self.compute_boundary(game_state)
        
    def compute_boundary(self, game_state):
        # Compute the middle boundary positions
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        
        boundary = []
        mid_x = width // 2
        
        if self.team == 'red':
            # Red team: boundary is just left of center
            boundary_x = mid_x - 1
        else:
            # Blue team: boundary is at center
            boundary_x = mid_x
            
        for y in range(1, height - 1):  # Skip border walls
            if not walls[boundary_x][y]:
                boundary.append((boundary_x, y))
                
        return boundary

    def get_closest_boundary(self, pos):
        # Find closest boundary position
        if not self.boundary:
            return None
        return min(self.boundary, key=lambda p: self.get_maze_distance(pos, p))

    def get_visible_enemies(self, game_state):
        # Get enemies with known positions
        visible = []
        for enemy in self.get_opponents(game_state):
            pos = game_state.get_agent_position(enemy)
            if pos is not None:
                visible.append(enemy)
        return visible

    def get_successor(self, game_state, action):
        # Get successor state
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_position(self.index)
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        return successor

    def is_in_own_territory(self, game_state, agent_index=None):
        #Check if agent is in own territory
        if agent_index is None:
            agent_index = self.index
            
        pos = game_state.get_agent_position(agent_index)
        if not pos:
            return True
            
        mid_x = game_state.data.layout.width // 2
        if self.team == 'red':
            return pos[0] < mid_x
        else:
            return pos[0] >= mid_x

    def choose_action(self, game_state):
        legal = game_state.get_legal_actions(self.index)
        if not legal:
            return Directions.STOP
            
        # Get action scores
        action_scores = []
        for action in legal:
            features = self.get_features(game_state, action)
            weights = self.get_weights(game_state, action)
            score = features * weights
            action_scores.append((score, action))
            
        # Choose best action
        best_score = max(action_scores)[0]
        best_actions = [a for s, a in action_scores if s == best_score]
        return random.choice(best_actions)

    def get_features(self, game_state, action):
        return util.Counter()
        
    def get_weights(self, game_state, action):
        return util.Counter()


class OffensiveAgent(BaseCaptureAgent):
    """
     offensive agent
    """

    def __init__(self, index):
        super().__init__(index)
        self.last_food_target = None
        self.food_history = []
        self.recent_positions = []

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        legal_actions = game_state.get_legal_actions(self.index)
        
        # Convert position to integer for wall checks
        my_pos_int = (int(my_pos[0]), int(my_pos[1]))
        
        # Remove STOP from legal actions unless it's the only option
        if len(legal_actions) > 1 and Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        # Track recent positions for stuck detection
        self.recent_positions.append(my_pos_int)
        if len(self.recent_positions) > 8:
            self.recent_positions.pop(0)

        # Get current food list
        food_list = self.get_food(game_state).as_list()
        
        #If carrying 5 or more, immediately return home
        if my_state.num_carrying >= 5:
            boundary = self.get_closest_boundary(my_pos_int)
            if boundary:
                best_action = None
                best_distance = float('inf')
                
                for action in legal_actions:
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                    dist = self.get_maze_distance(new_pos_int, boundary)
                    if dist < best_distance:
                        best_distance = dist
                        best_action = action
                
                if best_action:
                    return best_action

        # Adaptive threat distance based on food carried
        threat_multiplier = 1.0 + (my_state.num_carrying * 0.3)
        base_threat_distance = 3
        adaptive_threat_distance = base_threat_distance * threat_multiplier
        
        immediate_threat = self.has_immediate_threat(game_state, my_pos_int, adaptive_threat_distance)
        
        # Prioritize eating adjacent food if safe
        adjacent_food_actions = []
        for action in legal_actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            new_pos_int = (int(new_pos[0]), int(new_pos[1]))
            if new_pos_int in food_list:
                adjacent_food_actions.append(action)
        
        # If we have actions that lead directly to food, take one unless threat is too high
        if adjacent_food_actions:
            # With more food, we're more cautious about threats
            cautious_threat_distance = 2 + (my_state.num_carrying * 0.5)
            should_eat = not self.has_immediate_threat(game_state, my_pos_int, cautious_threat_distance)
            
            if should_eat:
                return random.choice(adjacent_food_actions)

        # Escape from threats (more sensitive with more food)
        if immediate_threat:
            # Escape to closest boundary
            best_action = None
            best_distance = float('inf')
            for action in legal_actions:
                successor = self.get_successor(game_state, action)
                new_pos = successor.get_agent_state(self.index).get_position()
                new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                boundary = self.get_closest_boundary(new_pos_int)
                if boundary:
                    dist = self.get_maze_distance(new_pos_int, boundary)
                    if dist < best_distance:
                        best_distance = dist
                        best_action = action
            if best_action:
                return best_action

        # amount of food to trigger return home
        base_return_threshold = 3
        
        # Calculate how deep we are in enemy territory
        boundary = self.get_closest_boundary(my_pos_int)
        territory_depth = 0
        if boundary:
            territory_depth = self.get_maze_distance(my_pos_int, boundary)
        
        # Adjust threshold based on depth - if we're deep, return sooner
        depth_adjustment = max(0, (territory_depth - 5) * 0.2)
        adjusted_threshold = max(1, base_return_threshold - depth_adjustment)  # Minimum threshold of 1
        
        if my_state.num_carrying >= adjusted_threshold:
            if boundary:
                best_action = None
                best_distance = float('inf')
                
                for action in legal_actions:
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                    dist = self.get_maze_distance(new_pos_int, boundary)
                    if dist < best_distance:
                        best_distance = dist
                        best_action = action
                
                if best_action:
                    return best_action

        # target food - But avoid risky foods when carrying a lot
        if food_list:
            # Filter out foods that are too risky when carrying a lot
            safe_foods = []
            risky_foods = []
            
            for food in food_list:
                # Check if food is in a safe position (close to boundary)
                food_risk = self.calculate_food_risk(food, my_state.num_carrying)
                risk_tolerance = max(0.5, 2.0 - (my_state.num_carrying * 0.3))  # Minimum tolerance of 0.5
                if food_risk <= risk_tolerance:
                    safe_foods.append(food)
                else:
                    risky_foods.append(food)
            
            # Prefer safe foods, but use risky ones if no safe options
            target_foods = safe_foods if safe_foods else risky_foods
            
            if target_foods:
                # Find closest food from target list
                closest_food = None
                closest_distance = float('inf')
                
                for food in target_foods:
                    dist = self.get_maze_distance(my_pos_int, food)
                    if dist < closest_distance:
                        closest_distance = dist
                        closest_food = food
            
                if closest_food:
                    # Find action that minimizes distance to this food
                    best_action = None
                    best_distance = float('inf')
                    
                    for action in legal_actions:
                        successor = self.get_successor(game_state, action)
                        new_pos = successor.get_agent_state(self.index).get_position()
                        new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                        dist = self.get_maze_distance(new_pos_int, closest_food)
                        if dist < best_distance:
                            best_distance = dist
                            best_action = action
                    
                    if best_action:
                        return best_action

        # chasae scared ghosts - Only if we're not carrying much
        if my_state.num_carrying < 2:
            scared_ghosts = self.get_scared_ghosts(game_state)
            if scared_ghosts:
                closest_scared = min(scared_ghosts, key=lambda g: self.get_maze_distance(my_pos_int, g.get_position()))
                ghost_dist = self.get_maze_distance(my_pos_int, closest_scared.get_position())
                if ghost_dist <= 4:
                    best_action = None
                    best_distance = float('inf')
                    
                    for action in legal_actions:
                        successor = self.get_successor(game_state, action)
                        new_pos = successor.get_agent_state(self.index).get_position()
                        new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                        dist = self.get_maze_distance(new_pos_int, closest_scared.get_position())
                        if dist < best_distance:
                            best_distance = dist
                            best_action = action
                    
                    if best_action:
                        return best_action

        # eat capsules - More important when carrying food
        capsules = self.get_capsules(game_state)
        if capsules and (self.has_nearby_threat(game_state, my_pos_int) or my_state.num_carrying >= 2):
            closest_capsule = min(capsules, key=lambda c: self.get_maze_distance(my_pos_int, c))
            capsule_dist = self.get_maze_distance(my_pos_int, closest_capsule)
            if capsule_dist <= 5:
                best_action = None
                best_distance = float('inf')
                
                for action in legal_actions:
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                    dist = self.get_maze_distance(new_pos_int, closest_capsule)
                    if dist < best_distance:
                        best_distance = dist
                        best_action = action
                
                if best_action:
                    return best_action

        # fallback: Use simple pathfinding to any food
        if food_list and my_state.num_carrying < 4:
            for food in food_list[:10]:
                best_action = None
                best_distance = float('inf')
                
                for action in legal_actions:
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                    dist = self.get_maze_distance(new_pos_int, food)
                    if dist < best_distance:
                        best_distance = dist
                        best_action = action
                
                if best_action and best_distance < 20:
                    return best_action

        # If we're carrying a lot and no food targets, just return home
        if my_state.num_carrying >= 2:
            boundary = self.get_closest_boundary(my_pos_int)
            if boundary:
                best_action = None
                best_distance = float('inf')
                
                for action in legal_actions:
                    successor = self.get_successor(game_state, action)
                    new_pos = successor.get_agent_state(self.index).get_position()
                    new_pos_int = (int(new_pos[0]), int(new_pos[1]))
                    dist = self.get_maze_distance(new_pos_int, boundary)
                    if dist < best_distance:
                        best_distance = dist
                        best_action = action
                
                if best_action:
                    return best_action

        # fallback: random legal action
        return random.choice(legal_actions)

    def has_immediate_threat(self, game_state, my_pos, threat_distance=3):
        #Check if there's an immediate threat
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        active_ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer <= 3]
        
        for ghost in active_ghosts:
            ghost_pos = ghost.get_position()
            ghost_pos_int = (int(ghost_pos[0]), int(ghost_pos[1]))
            if ghost_pos and self.get_maze_distance(my_pos, ghost_pos_int) <= threat_distance:
                return True
        return False

    def has_nearby_threat(self, game_state, my_pos):
        #Check if there's a nearby threat (ghost within 6 steps)
        return self.has_immediate_threat(game_state, my_pos, 6)

    def get_scared_ghosts(self, game_state):
        #Get scared ghosts that can be eaten
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        return [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer > 3]

    def calculate_food_risk(self, food_pos, num_carrying):
        # Calculate how risky a food position is (higher = more risky)
        # Foods closer to the boundary are safer
        boundary = self.get_closest_boundary(food_pos)
        if not boundary:
            return 2.0
            
        distance_to_boundary = self.get_maze_distance(food_pos, boundary)
        
        # Risk increases with distance from boundary
        risk = max(0, (distance_to_boundary - 3) * 0.2)
        
        return risk


class DefensiveAgent(BaseCaptureAgent):
    """
     defensive agent 
    """

    def __init__(self, index):
        super().__init__(index)
        self.state = "PATROL"
        self.patrol_target = None
        self.intercept_target = None
        self.stolen_foods = []
        self.last_food_count = None
        self.intercept_phase = "BOUNDARY"  # BOUNDARY or FOOD_LOCATION

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.last_food_count = len(self.get_food_you_are_defending(game_state).as_list())
        self.patrol_target = self.compute_optimal_patrol_position(game_state)

    def compute_optimal_patrol_position(self, game_state):
        #Find position that minimizes maximum distance to all foods
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if not food_list:
            return self.boundary[len(self.boundary) // 2] if self.boundary else self.start

        candidate_positions = self.get_defensive_positions(game_state)
        
        if not candidate_positions:
            return self.boundary[len(self.boundary) // 2] if self.boundary else self.start

        best_position = None
        best_max_distance = float('inf')

        for candidate in candidate_positions:
            max_dist = 0
            for food in food_list:
                dist = self.get_maze_distance(candidate, food)
                if dist > max_dist:
                    max_dist = dist
                    
            if max_dist < best_max_distance:
                best_max_distance = max_dist
                best_position = candidate

        return best_position if best_position else candidate_positions[0]

    def get_defensive_positions(self, game_state):
        #Get valid defensive positions (our territory, near boundary)
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        
        positions = []
        if self.team == 'red':
            x_range = range((width // 2) - 3, (width // 2))
        else:
            x_range = range((width // 2) + 1, (width // 2) + 4)
            
        for x in x_range:
            for y in range(1, height - 1):
                if not walls[x][y]:
                    positions.append((x, y))
                    
        return positions

    def update_state(self, game_state):
        #Update agent state based on current game situation
        my_state = game_state.get_agent_state(self.index)
        
        # Check for invaders
        invaders = self.get_invaders(game_state)
        
        # Update stolen food tracking
        self.update_stolen_foods(game_state)
        
        # State transitions
        if my_state.scared_timer > 0:
            # When scared, avoid invaders but still try to intercept
            if invaders and self.intercept_target is None:
                self.state = "PATROL"
            elif self.intercept_target:
                self.state = "INTERCEPT"
            else:
                self.state = "PATROL"
                
        elif invaders:
            # Not scared and invaders present - CHASE
            self.state = "CHASE"
            self.intercept_target = None
            self.intercept_phase = "BOUNDARY"
            
        elif self.intercept_target:
            # No invaders but we have stolen food to intercept
            self.state = "INTERCEPT"
            
        else:
            # No threats - patrol
            self.state = "PATROL"
            
        # Update patrol target periodically
        if random.random() < 0.05:
            self.patrol_target = self.compute_optimal_patrol_position(game_state)

    def get_invaders(self, game_state):
        #Get enemy pacmans in our territory
        invaders = []
        enemies = self.get_opponents(game_state)
        for enemy in enemies:
            enemy_state = game_state.get_agent_state(enemy)
            if enemy_state.is_pacman and enemy_state.get_position() is not None:
                if self.is_in_own_territory(game_state, enemy):
                    invaders.append(enemy)
        return invaders

    def update_stolen_foods(self, game_state):
        #Track stolen foods and set intercept targets
        current_food = self.get_food_you_are_defending(game_state).as_list()
        current_count = len(current_food)
        
        # Detect newly stolen food
        if self.last_food_count is not None and current_count < self.last_food_count:
            old_food_set = set(self.get_food_you_are_defending(game_state).as_list() + 
                             [f for f in self.stolen_foods if f not in current_food])
            new_food_set = set(current_food)
            stolen = list(old_food_set - new_food_set)
            
            for food_pos in stolen:
                if food_pos not in self.stolen_foods:
                    self.stolen_foods.append(food_pos)
                    
        self.last_food_count = current_count
        
        # Clean up intercepted foods when we reach them
        if self.stolen_foods and self.intercept_target:
            my_pos = game_state.get_agent_position(self.index)
            if my_pos and self.get_maze_distance(my_pos, self.intercept_target) <= 1:
                if self.intercept_target in self.stolen_foods:
                    self.stolen_foods.remove(self.intercept_target)
                self.intercept_target = None
                self.intercept_phase = "BOUNDARY"
        
        # Set new intercept target if needed
        if not self.intercept_target and self.stolen_foods:
            recent_food = self.stolen_foods[-1]
            crossing = self.find_best_intercept_point(game_state, recent_food)
            if crossing:
                self.intercept_target = crossing
                self.intercept_phase = "BOUNDARY"

    def find_best_intercept_point(self, game_state, food_pos):
        #Find the best boundary position to intercept enemy
        if not self.boundary:
            return None
        return min(self.boundary, key=lambda p: self.get_maze_distance(food_pos, p))

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Stay in own territory
        if not self.is_in_own_territory(successor):
            features['outside_territory'] = 1

        # Update state machine
        self.update_state(game_state)

        # Enhanced interception logic
        if self.state == "INTERCEPT" and self.intercept_target and self.stolen_foods:
            my_pos = successor.get_agent_state(self.index).get_position()
            
            # If we're at the boundary point and no enemies visible, go to the food location
            if self.intercept_phase == "BOUNDARY":
                if my_pos and self.get_maze_distance(my_pos, self.intercept_target) <= 1:
                    # Check if any invaders are visible
                    invaders = self.get_invaders(successor)
                    if not invaders:
                        # Switch to going to the actual food location
                        recent_food = self.stolen_foods[-1]
                        self.intercept_target = recent_food
                        self.intercept_phase = "FOOD_LOCATION"
            
            # Calculate distance to current intercept target
            if my_pos:
                features['intercept_distance'] = self.get_maze_distance(my_pos, self.intercept_target)

        # State-specific features
        if self.state == "CHASE":
            features.update(self.get_chase_features(successor, my_pos))
        elif self.state == "INTERCEPT":
            # Intercept features are already handled above
            pass
        else:  # PATROL
            features.update(self.get_patrol_features(successor, my_pos))

        # Penalties
        if action == Directions.STOP:
            features['stop'] = 1
            
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_chase_features(self, successor, my_pos):
       # Features for chasing invaders
        features = util.Counter()
        invaders = self.get_invaders(successor)
        
        if invaders:
            invader_dists = []
            for invader in invaders:
                invader_pos = successor.get_agent_position(invader)
                if invader_pos:
                    dist = self.get_maze_distance(my_pos, invader_pos)
                    invader_dists.append(dist)
                    
            if invader_dists:
                features['invader_distance'] = min(invader_dists)
                features['invader_count'] = len(invaders)
                
                if min(invader_dists) <= 1:
                    features['catch_invader'] = 1
                    
        return features

    def get_patrol_features(self, successor, my_pos):
        #Features for patrolling
        features = util.Counter()
        
        if self.patrol_target:
            features['patrol_distance'] = self.get_maze_distance(my_pos, self.patrol_target)
            
        food_list = self.get_food_you_are_defending(successor).as_list()
        if food_list:
            avg_dist = sum(self.get_maze_distance(my_pos, f) for f in food_list) / len(food_list)
            features['avg_food_distance'] = avg_dist
            
        return features

    def get_weights(self, game_state, action):
        #Weights based on current state
        base_weights = {
            'outside_territory': -1000.0,
            'stop': -50.0,
            'reverse': -10.0
        }
        
        if self.state == "CHASE":
            chase_weights = {
                'invader_count': -1000.0,
                'invader_distance': -20.0,
                'catch_invader': 500.0,
                'intercept_distance': 0.0,
                'patrol_distance': 0.0,
                'avg_food_distance': 0.0
            }
            base_weights.update(chase_weights)
            
        elif self.state == "INTERCEPT":
            intercept_weights = {
                'invader_count': -1000.0,
                'invader_distance': 0.0,
                'catch_invader': 0.0,
                'intercept_distance': -15.0,
                'patrol_distance': 0.0,
                'avg_food_distance': 0.0
            }
            base_weights.update(intercept_weights)
            
        else:  # PATROL
            patrol_weights = {
                'invader_count': -1000.0,
                'invader_distance': 0.0,
                'catch_invader': 0.0,
                'intercept_distance': 0.0,
                'patrol_distance': -8.0,
                'avg_food_distance': -2.0
            }
            base_weights.update(patrol_weights)
            
        # Adjust for scared state
        my_state = game_state.get_agent_state(self.index)
        if my_state.scared_timer > 0:
            if 'invader_distance' in base_weights and base_weights['invader_distance'] < 0:
                base_weights['invader_distance'] = 5.0
            base_weights['catch_invader'] = 0.0
            
        return base_weights