# baseline_team.py
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


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point

# This file contains two simple reflex-based capture agents used for the
# contest exercises: OffensiveReflexAgent and DefensiveReflexAgent.
# The base class ReflexCaptureAgent provides convenience helpers and a
# compact fallback choose_action implementation.  Each subclass implements
# role-specific decision logic (and is intended to be self-contained).


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # High-level: compute evaluation values for each legal action and
        # choose one of the actions with maximum value. Subclasses override
        # this behaviour to implement role-specific policies (offense/defense).
        actions = game_state.get_legal_actions(self.index)

        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        # The framework sometimes returns positions half-way between grid
        # points. To simplify decision logic we return the successor that
        # corresponds to an integer grid position (generate a second
        # successor if needed).
        
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        # Build a feature vector and compute a linear score using weights.
        # Subclasses extend get_features/get_weights to implement custom logic.
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Return a Counter with features for the base agent evaluation.

        This default implementation only provides a 'successor_score' feature
        (the team's score in the successor state). Subclasses extend this
        to add domain-specific features (food distance for offense, invader
        counts for defense, etc.).
        """
        
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    
    def choose_action(self, game_state):
        # Offensive agent decision procedure (prioritized rules)
        # Priority overview (high -> low):
        #  1) If carrying food and time is short -> return home
        #  2) Chase reachable scared ghosts (worthwhile to eat)
        #  3) Take a nearby power capsule (safe to pick)
        #  4) If a dangerous ghost is very close -> retreat
        #  5) If on home side and enemies visible -> choose the safest on-home maneuver
        #  6) If stuck at the border -> pick an alternate food target or escape deeper into home side
        #  7) Fallback: choose best evaluated action
        
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        # Safety policy for offense: carry/time/scared ghosts/capsules/retreat as implemented
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        
        # Observe opponents in this observation. 'get_position()' is None
        # for opponents we cannot see (out of sight range) — we only act on
        # visible agents. Split visible opponents into non-scared ghosts
        # (dangerous) and scared ones (edible targets).
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        visible_danger = [e for e in enemies if e.get_position() is not None and (not e.is_pacman) and getattr(e, 'scared_timer', 0) == 0]
        visible_scared = [e for e in enemies if e.get_position() is not None and getattr(e, 'scared_timer', 0) > 0]

        # midx: the vertical midline x-coordinate separating teams
        midx = game_state.data.layout.width // 2
        # Helper: true if a given position is on our side of the board
        def on_own_side(pos):
            if pos is None: return False
            return (pos[0] < midx) if self.red else (pos[0] >= midx)

        CHASE_THRESH = 5
        LEAVE_SAFE_DIST = 6

########################################################################
        # 1. If we're carrying food and time is almost up, head home so we deposit it
########################################################################

        carry = getattr(my_state, 'num_carrying', 0)
        remaining = getattr(game_state.data, 'timeleft', None)
        
        if carry > 0 and my_pos is not None and (not on_own_side(my_pos)) and remaining is not None:
            dist_home = self.get_maze_distance(my_pos, self.start)
            
            if (dist_home + 3) >= remaining:
                best_act = None
                best_dist = 99999
                
                for a in actions:
                    succ = self.get_successor(game_state, a)
                    pos = succ.get_agent_position(self.index)
                    if pos is None: continue
                    if on_own_side(pos):
                        return a
                    d = self.get_maze_distance(pos, self.start)
                    if d < best_dist:
                        best_dist = d
                        best_act = a
                        
                if best_act:
                    return best_act

########################################################################
        # 2. If one or more ghosts are scared, compute which scared ghosts are reachable
        # before their scared timer expires and are within a small radius; chase the closest one.
########################################################################

        if my_pos is not None and visible_scared:
            reachable = []
            for e in visible_scared:
                dist = self.get_maze_distance(my_pos, e.get_position())
                st = getattr(e, 'scared_timer', 0)
                if dist <= st and dist <= 4:
                    reachable.append((dist, e))
                    
            if reachable:
                # reachable is list of (distance, AgentState) — pick by distance only
                target = min(reachable, key=lambda t: t[0])[1]
                best_act = None
                best_d = 99999
                for a in actions:
                    succ = self.get_successor(game_state, a)
                    pos = succ.get_agent_position(self.index)
                    if pos is None: continue
                    d = self.get_maze_distance(pos, target.get_position())
                    if d < best_d:
                        best_d = d
                        best_act = a
                if best_act:
                    return best_act
                
########################################################################            
        # 3. Take a nearby power capsule (safe to pick)
        # only pick up capsules when there are no immediate visible enemies.
########################################################################

        CAPS_THRESH = 3
        capsules = self.get_capsules(game_state)
        
        if my_pos is not None and capsules and (not visible_danger):
            nearest = min(capsules, key=lambda c: self.get_maze_distance(my_pos, c))
            
            if self.get_maze_distance(my_pos, nearest) <= CAPS_THRESH:
                best_act = None
                best_d = 99999
                
                for a in actions:
                    succ = self.get_successor(game_state, a)
                    pos = succ.get_agent_position(self.index)
                    if pos is None: continue
                    d = self.get_maze_distance(pos, nearest)
                    if d < best_d:
                        best_d = d
                        best_act = a
                        
                if best_act:
                    return best_act
                
########################################################################            
        # 4. If a dangerous (non-scared) ghost is within CHASE_THRESH steps -> retreat
########################################################################            
   
        if my_pos is not None and any(self.get_maze_distance(my_pos, e.get_position()) <= CHASE_THRESH for e in visible_danger):
            best_act = None
            best_dist = 99999
            
            for a in actions:
                succ = self.get_successor(game_state, a)
                pos = succ.get_agent_position(self.index)
                if on_own_side(pos):
                    return a
                if pos is None: continue
                d = self.get_maze_distance(pos, self.start)
                if d < best_dist:
                    best_dist = d
                    best_act = a
                    
            if best_act:
                return best_act

########################################################################
        # 5. If staying or returning home (i.e., on our own side),
        # keep distance from visible enemies
########################################################################

        if my_pos is not None and on_own_side(my_pos) and visible_danger:
            best_act = None
            best_min = -1
            rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            
            for a in actions:
                succ = self.get_successor(game_state, a)
                pos = succ.get_agent_position(self.index)
                if pos is None: continue
                if not on_own_side(pos):
                    closest_to_border = min(self.get_maze_distance(e.get_position(), (midx, pos[1])) for e in visible_danger)
                    if closest_to_border < LEAVE_SAFE_DIST:
                        continue
                dists = [self.get_maze_distance(pos, e.get_position()) for e in visible_danger]
                min_d = min(dists) if dists else 9999
                if min_d > best_min:
                    best_min = min_d
                    best_act = a
                    
            # If there's a safe action, take it
            if best_act:
                # avoid selecting stop or immediate reverse if possible
                if best_act == Directions.STOP or best_act == rev:
                    # find alternative on-home maneuver that moves us (not stop/reverse)
                    maneuvers = []
                    
                    for a in actions:
                        if a == Directions.STOP or a == rev: continue
                        succ = self.get_successor(game_state, a)
                        pos = succ.get_agent_position(self.index)
                        if pos is None: continue
                        if not on_own_side(pos): continue
                        dists = [self.get_maze_distance(pos, e.get_position()) for e in visible_danger]
                        min_d = min(dists) if dists else 9999
                        maneuvers.append((min_d, a))
                        
                    if maneuvers:
                        return max(maneuvers)[1]
                    
                return best_act

########################################################################
        # 7. Come back home if pacman already ate 18 or more food dots
########################################################################

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        # Offensive agent feature extractor: encourage collecting fewer food
        # (the successor_score) and prefer shorter distance to nearest food.
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        # Weights for the offensive evaluator: strongly prefer reducing the
        # number of food pellets and slightly prefer closer food.
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        # Defensive feature set: whether we're on defense, number of visible
        # invaders and distance to the nearest invader; when no invaders are
        # visible we also include distance to the centroid of defended food.
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # When there are no invaders, prefer to stay near the "mean" location
        # of the food we are defending (centroid of remaining food on our side).
        # We slightly move that centroid toward the border so the defender stays
        # closer to the boundary where enemies typically cross.
        if len(invaders) == 0:
            food_list = self.get_food_you_are_defending(successor).as_list()
            if food_list and my_pos is not None:
                mx = sum(p[0] for p in food_list) / len(food_list)
                my = sum(p[1] for p in food_list) / len(food_list)
                centroid = (int(round(mx)), int(round(my)))
                
                # move centroid toward the border (midx) so defender patrols nearer the midline
                alpha = 0.6
                midx = successor.data.layout.width // 2
                border_x = midx - 1 if self.red else midx
                adj_x = int(round(centroid[0] * (1 - alpha) + border_x * alpha))
                if self.red and adj_x >= midx:
                    adj_x = midx - 1
                if (not self.red) and adj_x < midx:
                    adj_x = midx
                centroid = (adj_x, centroid[1])
                try:
                    features['dist_to_center'] = self.get_maze_distance(my_pos, centroid)
                except Exception:
                    # if dist calculation fails for any reason, fall back to 0
                    features['dist_to_center'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        # Defensive weights: discourage invaders and prefer being on defense.
        # invader_distance is negative to bring the agent closer to visible invaders.
        # dist_to_center is modest to keep the defender near the centroid of food.
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'dist_to_center': -1, 'stop': -100, 'reverse': -2}

    def choose_action(self, game_state):
        """Deterministic defensive policy:
        - If invader(s) visible: choose action that minimizes maze distance to the nearest invader.
        - Otherwise: move toward centroid (mean) of the food you are defending (keeps defender near the bulk of dots).
        Avoid Stop or immediate Reverse when possible to reduce stalling.
        """
        
        actions = game_state.get_legal_actions(self.index)
        if not actions:
            return None

        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Current facing direction's reverse: we generally avoid taking a
        # direct reverse in a single step (it often indicates back-and-forth
        # stalling). Keep 'rev' handy for action tie-breaking.
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]

        opponents = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in opponents if a.is_pacman and a.get_position() is not None]

        # If we see invaders, pick the action that reduces the maze distance
        # to the nearest invader.
        if my_pos is not None and invaders:
            best = None
            best_d = float('inf')
            for a in actions:
                succ = self.get_successor(game_state, a)
                pos = succ.get_agent_position(self.index)
                if pos is None: continue
                d = min(self.get_maze_distance(pos, inv.get_position()) for inv in invaders)
                # prefer non-stop/non-reverse actions when distances tie
                tie_penalty = 0
                if a == Directions.STOP: tie_penalty += 1
                if a == rev: tie_penalty += 0.5
                val = (d, tie_penalty)
                if val < (best_d, 999):
                    best_d, _ = val
                    best = a
            if best:
                return best

        # No visible invaders: move toward the centroid of defended food.
        # This heuristic places the defender near the area that contains the
        # majority of remaining food, improving coverage. The centroid is
        # nudged toward the border so the defender sits closer to likely
        # intruder crossing points.
        food_list = self.get_food_you_are_defending(game_state).as_list()
        if my_pos is not None and food_list:
            mx = sum(p[0] for p in food_list) / len(food_list)
            my = sum(p[1] for p in food_list) / len(food_list)
            centroid = (int(round(mx)), int(round(my)))
            
            # shift centroid toward the border to keep defender nearer the midline
            alpha = 0.6
            midx = game_state.data.layout.width // 2
            border_x = midx - 1 if self.red else midx
            adj_x = int(round(centroid[0] * (1 - alpha) + border_x * alpha))
            if self.red and adj_x >= midx:
                adj_x = midx - 1
            if (not self.red) and adj_x < midx:
                adj_x = midx
            centroid = (adj_x, centroid[1])
            best = None
            best_d = float('inf')
            
            for a in actions:
                # prefer staying on our side
                succ = self.get_successor(game_state, a)
                pos = succ.get_agent_position(self.index)
                if pos is None: continue
                
                # ensure we don't leave our side accidentally
                midx = game_state.data.layout.width // 2
                on_own = (pos[0] < midx) if self.red else (pos[0] >= midx)
                if not on_own:
                    continue
                try:
                    d = self.get_maze_distance(pos, centroid)
                except Exception:
                    # centroid might be a wall or invalid position; fall back to nearest defended food
                    d = min(self.get_maze_distance(pos, f) for f in food_list)
                    
                # avoid stop/reverse if there are alternatives
                penalty = 0
                if a == Directions.STOP: penalty += 1
                if a == rev: penalty += 0.5
                val = (d, penalty)
                if val < (best_d, 999):
                    best_d, _ = val
                    best = a
            if best:
                return best

        # Fallback: default behavior (use base evaluation)
        return super().choose_action(game_state)
