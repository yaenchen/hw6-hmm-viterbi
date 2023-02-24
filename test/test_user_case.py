"""
UCSF BMI203: Biocomputing Algorithms
Author: Yaen Chen
Date: 2/24/2023
Program: BMI
Description: Pytest for user cases.
"""
import pytest
import numpy as np
from src.models.hmm import HiddenMarkovModel
from src.models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """
    Evaluating if HMM can predict if a grad student is committed or ambivalent to their rotation lab based on PI funding hidden states.
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm
    # Check HMM dimensions and ViterbiAlgorithm
    # transition and emission probs should have same dimensions
    assert np.shape(use_case_one_viterbi.hmm_object.transition_probabilities) == np.shape(use_case_one_viterbi.hmm_object.emission_probabilities)
    # hidden states and observed states should also have same dimensions
    assert np.shape(use_case_one_viterbi.hmm_object.hidden_states) == np.shape(use_case_one_viterbi.hmm_object.observation_states)
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """
    Observing if a HMM can predict if an individual will be late or on time depending on traffic states.
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm
    # transition and emission probs should have same dimensions
    assert np.shape(use_case_one_viterbi.hmm_object.transition_probabilities) == np.shape(use_case_one_viterbi.hmm_object.emission_probabilities)
    # hidden states and observed states should also have same dimensions
    assert np.shape(use_case_one_viterbi.hmm_object.hidden_states) == np.shape(use_case_one_viterbi.hmm_object.observation_states)

    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_two():
    """
    Observing whether humidity levels predict whether there will be rain or if it will be sunny.
    We expect there to be no relationship between the hidden states and observed states.
    """
    # create a case where an HMM would not be appropriate. There is no hidden state pattern underlying observations.
    weather_data = np.load('./data/PersonalCase.npz')
    # index annotation observation_states=[i,j]
    observation_states = ['rain', 'sunny']

    # index annotation hidden_states=[i,j]
    hidden_states = ['high-humidity', 'low-humidity']

    weather_hmm_data = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      weather_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      weather_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      weather_data['emission_probabilities'])

    # Run Viterbi algorithm
    weather_viterbi = ViterbiAlgorithm(weather_hmm_data)
    # Find the best hidden state path for our observation states
    weather_viterbi_decoded_hidden_states = weather_viterbi.best_hidden_state_sequence(weather_data['observation_states'])

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert weather_viterbi.hmm_object.observation_states == weather_hmm_data.observation_states
    assert weather_viterbi.hmm_object.hidden_states == weather_hmm_data.hidden_states

    assert np.allclose(weather_viterbi.hmm_object.prior_probabilities, weather_hmm_data.prior_probabilities)
    assert np.allclose(weather_viterbi.hmm_object.transition_probabilities,
                       weather_hmm_data.transition_probabilities)
    assert np.allclose(weather_viterbi.hmm_object.emission_probabilities, weather_hmm_data.emission_probabilities)

    # The predicted hidden states should not match
    assert np.alltrue(weather_viterbi_decoded_hidden_states == weather_data['hidden_states']) == False

def test_user_case_three():
    """
        Observing if experiencing adverse childhood experiences (low, some, or high ACEs) are a hidden state that underlies
        whether an individual is upper, middle, or lower class as an adult using simulated data.
    """
    # load data
    aces_data = np.load('./data/PersonalCase2.npz')
    # index annotation observation_states=[i,j]
    observation_states = ['lower', 'middle', 'upper']

    # index annotation hidden_states=[i,j]
    hidden_states = ['high-ace', 'some-ace', 'low-ace']

    aces_hmm_data = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         aces_data['prior_probabilities'],
                                         # prior probabilities of hidden states in the order specified in the hidden_states list
                                         aces_data['transition_probabilities'],
                                         # transition_probabilities[:,hidden_states[i]]
                                         aces_data['emission_probabilities'])

    # Run Viterbi algorithm
    aces_viterbi = ViterbiAlgorithm(aces_hmm_data)
    # Find the best hidden state path for our observation states
    aces_viterbi_decoded_hidden_states = aces_viterbi.best_hidden_state_sequence(
        aces_data['observation_states'])

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert aces_viterbi.hmm_object.observation_states == aces_hmm_data.observation_states
    assert aces_viterbi.hmm_object.hidden_states == aces_hmm_data.hidden_states

    assert np.allclose(aces_viterbi.hmm_object.prior_probabilities, aces_hmm_data.prior_probabilities)
    assert np.allclose(aces_viterbi.hmm_object.transition_probabilities,
                       aces_hmm_data.transition_probabilities)
    assert np.allclose(aces_viterbi.hmm_object.emission_probabilities, aces_hmm_data.emission_probabilities)

    # The predicted hidden states should match
    assert np.alltrue(aces_viterbi_decoded_hidden_states == aces_data['hidden_states'])
