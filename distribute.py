import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns

def vis(X, title:str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(X, annot=False, cmap='coolwarm', center=0)
    plt.title(f'Heatmap of {title} Matrix')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    plt.show()

def main():
    np.random.seed(33)
    # Step 1: Generate the Sparse Matrix X
    p = 40
    X = np.diag([8] * p)  # Tri-diagonal matrix with diagonal entries 8
    upper_tri_indices = np.triu_indices(p, k=1)  # Upper triangular indices
    num_upper_tri_nonzeros = 38

    # Select 38 random upper triangular positions
    chosen_indices = np.random.choice(range(len(upper_tri_indices[0])), num_upper_tri_nonzeros, replace=False)
    for idx in chosen_indices:
        i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        value = np.random.uniform(-2, 2)
        X[i, j] = value
        X[j, i] = value
    vis(X, "original")

    # Step 2: Generate Matrices A and B
    m = 21
    A = np.zeros((m, p))
    B = np.zeros((m, p))

    for i in range(p):
        A[np.random.choice(m, 4, replace=False), i] = 1
        B[np.random.choice(m, 4, replace=False), i] = 1

    # Step 3: Form the Measurement Matrix Y = AXB^T
    Y = A @ X @ B.T

    # Step 4: Solve the Optimization Problem (P1) to reconstruct X
    X_hat = cp.Variable((p, p))
    objective = cp.Minimize(cp.norm(X_hat, 1))
    constraints = [A @ X_hat @ B.T == Y]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Extract the estimated X
    X_reconstructed = X_hat.value
    vis(X_reconstructed, "reconstruction")
    # np.savetxt("./out/original_X.txt", X, fmt="%.5f")
    # np.savetxt("./out/reconst_X.txt", X_reconstructed, fmt="%.5f")
    print(np.sum(np.abs(X - X_reconstructed)))

if __name__ == "__main__":
    main()
