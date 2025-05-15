import numpy as np

# Define values of X and Y
X_vals = np.array([0, 0, 1, 1])
Y_vals = np.array([0, 1, 0, 1])
P_joint = np.array([0.1, 0.4, 0.2, 0.3])  # P(X, Y)

# Marginalize over Y to get P(X)
P_X = np.zeros(2)  # since X in {0, 1}
for i in range(len(P_joint)):
    P_X[X_vals[i]] += P_joint[i]

print("Marginal distribution P(X):")
for x in [0, 1]:
    print(f"P(X={x}) = {P_X[x]}")

# Compute expected value of X
expected_X = np.sum(P_X * np.array([0, 1]))
print(f"\nExpected value E[X]: {expected_X}")
