
# it must calculate one time just in start of the optimization and passed through GAs

import numpy as np
from itertools import product

# Step values: 0.0 to 1.0 in steps of 0.1
def make_weights():

    step_values = [round(i * 0.1, 1) for i in range(11)]

    valid_genes = []


    # Generate all possible 8-gene combinations from the step_values
    for gene in product(step_values, repeat=8):
        if is_valid(gene):
            valid_genes.append(np.array(gene))

    return valid_genes



def is_valid(gene):
    return (
        abs(sum(gene) - 1.0) < 1e-6 and
        sum(gene[0:2]) >= 0.125 and
        sum(gene[2:5]) >= 0.125 and
        sum(gene[5:8]) >= 0.125 and
        all(g >= 0 for g in gene)
    )
