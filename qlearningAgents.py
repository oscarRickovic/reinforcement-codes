# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.qValues = {}  # Dictionnaire pour stocker les valeurs Q

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    "*** YOUR CODE HERE ***"
     # Renvoie la valeur Q pour un état et une action donnés
    # Initialisation à 0.0 si l'agent n'a jamais vu cet état-action
    return self.qValues.get((state, action), 0.0)


  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    "*** YOUR CODE HERE ***"
    # Obtenir toutes les actions légales pour cet état
    legalActions = self.getLegalActions(state)
    if not legalActions:  # Si aucune action n'est légale (état terminal)
        return 0.0
    # Retourner la valeur maximale parmi toutes les actions légales
    return max(self.getQValue(state, action) for action in legalActions)

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    "*** YOUR CODE HERE ***"
    # Obtenir toutes les actions légales pour cet état
    legalActions = self.getLegalActions(state)
    if not legalActions:  # Si aucune action n'est légale (état terminal)
        return None
    # Trouver les actions ayant la valeur Q maximale
    maxQValue = max(self.getQValue(state, action) for action in legalActions)
    # Filtrer les actions ayant la même valeur Q maximale
    bestActions = [action for action in legalActions if self.getQValue(state, action) == maxQValue]
    # Choisir une action au hasard parmi les meilleures actions
    return random.choice(bestActions)

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None
    "*** YOUR CODE HERE ***"
    # Obtenir toutes les actions légales pour cet état
    legalActions = self.getLegalActions(state)
    if not legalActions:  # Si aucune action n'est légale (état terminal)
        return None
    
    # Avec une probabilité epsilon, choisir une action aléatoire (exploration)
    if util.flipCoin(self.epsilon):
        return random.choice(legalActions)
    else:
        # Sinon, choisir l'action optimale (exploitation)
        return self.getPolicy(state)

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    "*** YOUR CODE HERE ***"
    # Obtenir la valeur maximale de Q pour l'état suivant
    maxNextQValue = self.getValue(nextState)
    
    # Calculer la nouvelle valeur de Q en utilisant l'équation du Q-learning
    currentQValue = self.getQValue(state, action)
    newQValue = currentQValue + self.alpha * (reward + self.discount * maxNextQValue - currentQValue)
    
    # Mettre à jour la valeur Q dans le dictionnaire
    self.qValues[(state, action)] = newQValue

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


from featureExtractors import IdentityExtractor, SimpleExtractor

class ApproximateQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        PacmanQAgent.__init__(self, **args)
        # Convert the extractor string to the actual class
        extractors = {
            'IdentityExtractor': IdentityExtractor,
            'SimpleExtractor': SimpleExtractor
        }
        self.featExtractor = extractors[extractor]()
        self.weights = util.Counter()

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        return sum([self.weights[feature] * value for feature, value in features.items()])

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(state, action)
        q_value = self.getQValue(state, action)
        next_q_value = self.getValue(nextState)
        difference = (reward + self.discount * next_q_value) - q_value
        for feature, value in features.items():
            self.weights[feature] += self.alpha * difference * value

