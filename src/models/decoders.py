import copy
import numpy as np


class ViterbiAlgorithm:
    """
    ViterbiAlgorithm will run the Viterbi Algorithm on a hmm_object, which is instantiated as a ViterbiAlgorithm class attribute.
    """

    def __init__(self, hmm_object):
        """
        Adds hmm_object as a ViterbiAlgorithm class attribute.

        Args:
            hmm_object (_type_): a HiddenMarkovModel class object.
        """
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """
        Runs the Viterbi Algorithm to determine the observation states based on transition, emission, and prior probabilities
        of hidden states, stored in hmm_object.

        Args:
            decode_observation_states (np.ndarray): observation states observed.

        Returns:
            np.ndarray: np.array containing the optimal observation states using hidden states and probabilities with the Viterbi Algorithm.
        """

        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states),
                         len(self.hmm_object.hidden_states)))
        path[0, :] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]

        best_path = np.zeros(len(decode_observation_states))

        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        delta = np.multiply(self.hmm_object.prior_probabilities, np.transpose(self.hmm_object.emission_probabilities[:,
                                                                              self.hmm_object.observation_states_dict.get(
                                                                                  decode_observation_states[0])]))
        # 2. Scale: normalize probability of transitioning to next observed state given transition/emission probs of hidden state
        # delta values sum to 1
        delta = delta / np.sum(delta)
        best_path[0] = path[0][np.argmax(delta)]
        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        # trellis is way of walking down the path of observed hidden states, the path that we take
        for trellis_node in range(1, len(decode_observation_states)):
            # product of delta and transition
            # then product between that and emission probability of the observation in the nth state of trellis path
            product_of_delta_and_transition_emission = np.multiply(delta,
                                                                   self.hmm_object.transition_probabilities.transpose())
            # emission: prob that hidden state is emitting the observed state. rows are hidden states and columns are obs
            product_of_delta_and_transition_emission = np.multiply(product_of_delta_and_transition_emission,
                                                                   self.hmm_object.emission_probabilities[:,
                                                                   self.hmm_object.observation_states_dict.get(
                                                                       decode_observation_states[trellis_node])])
            # maximize product_of_delta_and_transition_emission
            # max for each COLUMN (what each hidden state/obs state corresponds to), then transpose so
            max_prob_hidden_state = product_of_delta_and_transition_emission.max(axis=1).transpose()
            # obs state are rows now, hidden states are columns
            # scale probabilities
            scaled_max_probs = max_prob_hidden_state / np.sum(max_prob_hidden_state)
            # track indices of observed states (to decode hidden states) based on the max value we chose (rows are observed states in product_of_delta_and_transition_emission)
            obs_state_index = product_of_delta_and_transition_emission.argmax(axis=1)
            # add the observed indices to the path variable to keep track
            path[trellis_node] = obs_state_index
            # using our delta, get the next observed state based on the max probability
            best_path[trellis_node - 1] = path[trellis_node - 1][np.argmax(scaled_max_probs)]
            # recalculate delta
            delta = np.multiply(self.hmm_object.prior_probabilities, np.transpose(
                self.hmm_object.emission_probabilities[:,
                self.hmm_object.observation_states_dict.get(decode_observation_states[trellis_node])]))

        best_hidden_state_path = np.array([self.hmm_object.hidden_states[np.int32(index)] for index in best_path])
        return best_hidden_state_path