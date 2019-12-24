from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        Osequence_index = self.obs_dict[Osequence[0]]
        alpha[:, 0] = self.pi * self.B[:, Osequence_index]
        A_Transpose = np.transpose(self.A)
        t = 1
        while t < L:
            alpha[:, t] = np.multiply(np.dot(A_Transpose, alpha[:, t - 1].transpose()), self.B[:, self.obs_dict[Osequence[t]]])
            t = t + 1
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        beta[:, L - 1] = 1
        t = L - 2
        while t >= 0:
            beta[:, t] = np.dot(self.A, np.multiply(beta[:, t + 1], self.B[:, self.obs_dict[Osequence[t+1]]].transpose()))
            t = t - 1
        ###################################################
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        a = self.forward(Osequence)
        prob = np.sum(a[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        a = self.forward(Osequence)
        b = self.backward(Osequence)
        prob = np.multiply(a,b) / np.sum(a[:, -1])
        ###################################################
        return prob

    # TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([ S, S, L - 1])
        ###################################################
        a = self.forward(Osequence)
        b = self.backward(Osequence)
        x = np.multiply(a[:, np.newaxis,:-1] , b[:,1:])
        y = np.multiply(self.A[:,:,np.newaxis], self.B[:,[self.obs_dict[o] for o in Osequence[1:]]])
        prob = np.multiply(x,y) / np.sum(a[:, -1])
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        L = len(Osequence)
        S = len(self.pi)

        delta = np.zeros([S, L])

        for j in range(S):
            delta[j, 0] = self.pi[j] * self.B[j][self.obs_dict[Osequence[0]]]

        t = 1

        dp = np.zeros([S, L])
        A_transpose = self.A.transpose()
        while t < L:
            probability = np.multiply(delta[:, t- 1], A_transpose)
            max_probability = np.max(probability, axis = 1)
            delta[:, t] = np.multiply(max_probability, self.B[:,self.obs_dict[Osequence[t]]])
            dp[:, t] = np.argmax(probability, axis = 1)
            t = t + 1

        index = np.zeros(L)

        index[L-1] = np.argmax(delta[:, -1])

        t = L - 2

        while t >= 0:
            index[t] = dp[int(index[t + 1]), t + 1]
            t = t - 1

        states = dict([(v, k) for k, v in self.state_dict.items()])

        for i in index:
            path.append(states[i])

        ###################################################
        return path
