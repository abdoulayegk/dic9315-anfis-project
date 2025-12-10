"""
PyTorch implementation of ANFIS (Adaptive Neuro-Fuzzy Inference System)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianMF(nn.Module):
    """Gaussian Membership Function Layer"""
    def __init__(self, n_inputs, n_membership_functions):
        super(GaussianMF, self).__init__()
        self.n_inputs = n_inputs
        self.n_mf = n_membership_functions
        
        # Parameters: mu (mean) and sigma (standard deviation)
        # Shape: (n_inputs, n_mf)
        self.mu = nn.Parameter(torch.randn(n_inputs, n_membership_functions))
        self.sigma = nn.Parameter(torch.ones(n_inputs, n_membership_functions))
        
    def forward(self, x):
        # x shape: (batch_size, n_inputs)
        # Output shape: (batch_size, n_inputs, n_mf)
        
        # Expand x to match MF shape
        # x_expanded: (batch_size, n_inputs, 1)
        x_expanded = x.unsqueeze(2)
        
        # Calculate Gaussian membership
        # exponent = -0.5 * ((x - mu) / sigma) ** 2
        exponent = -0.5 * ((x_expanded - self.mu) / self.sigma.abs()) ** 2
        return torch.exp(exponent)

class ANFISNetwork(nn.Module):
    """
    Takagi-Sugeno ANFIS Network
     Optimized for 5 variables
    """
    def __init__(self, n_inputs, n_rules):
        super(ANFISNetwork, self).__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        
        # Layer 1: Fuzzification
        # We treat n_rules as the number of fuzzy sets per input for simplicity 
        # in this grid-like or cluster-like approach
        # For 5 inputs, full grid is too big, so we assume 1-to-1 mapping 
        # between rules and clusters (Subtractive Clustering approach)
        self.fuzzification = GaussianMF(n_inputs, n_rules)
        
        # Layer 4: Consequent (Takagi-Sugeno)
        # y = Ax + b for each rule
        # Weights shape: (n_rules, n_inputs)
        # Bias shape: (n_rules,)
        self.consequent_weights = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.consequent_bias = nn.Parameter(torch.randn(n_rules))
        
    def forward(self, x):
        # x: (batch_size, n_inputs)
        batch_size = x.shape[0]
        
        # --- Layer 1: Fuzzification ---
        # memberships: (batch_size, n_inputs, n_rules)
        # In this simplified architecture (cluster-based), 
        # we assume the k-th MF of each input belongs to the k-th rule.
        # This avoids the exponential explosion of a full grid.
        memberships = self.fuzzification(x)
        
        # --- Layer 2: Rule Firing Strength (T-Norm) ---
        # For cluster-based ANFIS, rule k uses the k-th MF of all inputs.
        # w[k] = mu_1[k] * mu_2[k] * ... * mu_n[k]
        # firing_strengths: (batch_size, n_rules)
        firing_strengths = torch.prod(memberships, dim=1)
        
        # --- Layer 3: Normalization ---
        # normalized_weights: (batch_size, n_rules)
        # Add epsilon to avoid division by zero
        sum_weights = firing_strengths.sum(dim=1, keepdim=True) + 1e-10
        normalized_weights = firing_strengths / sum_weights
        
        # --- Layer 4: Consequent ---
        # f_i = a_i * x + b_i
        # We need to compute this for each rule i and each sample in batch
        
        # x_expanded: (batch_size, 1, n_inputs)
        x_expanded = x.unsqueeze(1)
        
        # rule_outputs: (batch_size, n_rules)
        # (batch, 1, n_in) * (n_rules, n_in) -> sum over inputs
        rule_outputs = (x_expanded * self.consequent_weights).sum(dim=2) + self.consequent_bias
        
        # --- Layer 5: Aggregation ---
        # final_output = sum(norm_weight_i * rule_output_i)
        # output: (batch_size,)
        output = (normalized_weights * rule_outputs).sum(dim=1)
        
        return output, normalized_weights

    def get_rules(self):
        """Extract readable rules from the network"""
        rules = []
        mu = self.fuzzification.mu.detach().cpu().numpy()
        sigma = self.fuzzification.sigma.detach().cpu().numpy()
        
        # This is a placeholder for rule extraction logic
        return rules

