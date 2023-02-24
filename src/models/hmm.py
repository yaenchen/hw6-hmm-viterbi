import numpy as np
class HiddenMarkovModel:
    """Class to instantiate HiddenMarkovModel objects, which contain the observation/hidden states and prior/transition/emission probabilities
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """
        Initializes HiddenMarkovModel object

        Args:
            observation_states (np.ndarray): all possible observation states
            hidden_states (np.ndarray): all possible hidden states
            prior_probabilities (np.ndarray): prior probabilities of an observed state
            transition_probabilities (np.ndarray): transition probabilities contain probabilities that the hidden state will change from one to another. rows are observed states and columns are hidden states.
            emission_probabilities (np.ndarray): probabilities that the hidden state is emitting the observed state. rows are hidden states and columns are observed states.
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities = prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities