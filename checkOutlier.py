"""
Outlier Detection and Visualization

This script creates box plots to visualize outliers in the log return data.
Useful for identifying extreme values and understanding return distributions.

Usage: python checkOutlier.py

Output: Displays box plot (use matplotlib save button to save as PNG)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load log return data
df = pd.read_csv("logRetData.csv")

# Create box plot for all ETFs
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title('Box Plot of ETF Log Returns - Outlier Detection', fontsize=14, fontweight='bold')
plt.xlabel('ETF', fontsize=12)
plt.ylabel('Log Return', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
