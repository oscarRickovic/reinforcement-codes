# featureExtractors.py
# --------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:  
  def getFeatures(self, state, action):    
    """
      Returns a dict from features to counts
      Usually, the count will just be 1.0 for
      indicator functions.  
    """
    util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
  def getFeatures(self, state, action):
    feats = util.Counter()
    feats[(state,action)] = 1.0
    return feats

def closestFood(pos, food, walls):
  """
  closestFood -- this is similar to the function that we have
  worked on in the search project; here its all in one place
  """
  fringe = [(pos[0], pos[1], 0)]
  expanded = set()
  while fringe:
    pos_x, pos_y, dist = fringe.pop(0)
    if (pos_x, pos_y) in expanded:
      continue
    expanded.add((pos_x, pos_y))
    # if we find a food at this location then exit
    if food[pos_x][pos_y]:
      return dist
    # otherwise spread out from the location to its neighbours
    nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
    for nbr_x, nbr_y in nbrs:
      fringe.append((nbr_x, nbr_y, dist+1))
  # no food found
  return None

class SimpleExtractor(FeatureExtractor):
  """
  Returns enhanced features for a smarter Pac-Man:
  - Whether food will be eaten
  - How far away the next food is
  - Whether a ghost collision is imminent
  - Whether a ghost is one step away
  - Distance to the closest capsule
  - Detection of edible ghosts and their distances
  """
  
  def getFeatures(self, state, action):
      # Extract the grid of food and wall locations and get the ghost locations
      food = state.getFood()
      walls = state.getWalls()
      capsules = state.getCapsules()
      ghosts = state.getGhostStates()

      features = util.Counter()
      features["bias"] = 1.0

      # Compute the location of Pac-Man after the action
      x, y = state.getPacmanPosition()
      dx, dy = Actions.directionToVector(action)
      next_x, next_y = int(x + dx), int(y + dy)

      # Count the number of ghosts 1-step away
      ghost_positions = [ghost.getPosition() for ghost in ghosts]
      features["#-of-ghosts-1-step-away"] = sum(
          (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghost_positions
      )

      # Check for edible ghosts and their distances
      edible_ghosts = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
      if edible_ghosts:
          closest_edible_ghost_dist = min(
              util.manhattanDistance((next_x, next_y), ghost.getPosition())
              for ghost in edible_ghosts
          )
          features["closest-edible-ghost"] = float(closest_edible_ghost_dist) / (walls.width * walls.height)

      # Add feature for eating food
      if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
          features["eats-food"] = 1.0

      # Distance to the closest food
      dist = closestFood((next_x, next_y), food, walls)
      if dist is not None:
          features["closest-food"] = float(dist) / (walls.width * walls.height)

      # Distance to the closest capsule
      if capsules:
          closest_capsule_dist = min(
              util.manhattanDistance((next_x, next_y), c) for c in capsules
          )
          features["closest-capsule"] = float(closest_capsule_dist) / (walls.width * walls.height)

      # Avoid ghosts by default
      features["avoid-ghosts"] = 0.0
      if features["#-of-ghosts-1-step-away"] > 0:
          features["avoid-ghosts"] = 1.0

      # Normalize feature values
      features.divideAll(10.0)

      return features
