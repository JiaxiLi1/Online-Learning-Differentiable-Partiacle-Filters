import numpy as np


def generate_data(batch_size, sequence_length, ts=1.0, p0=1.0, alpha=2.0, r=np.array([0.0, 0.0])):
    # Initialize the data containers
    states = np.zeros((batch_size, sequence_length, 5))  # Including omega_t
    observations = np.zeros((batch_size, sequence_length, 2))  # Two observation components

    for batch in range(batch_size):
        # Initial state
        x_tilde = np.array([0.0, 0.0, 0.0, 0.0])  # Initial position and velocity
        omega_t = 0.0  # Initial turn rate

        for t in range(sequence_length):
            # Sample state noise u_t and turn rate noise u_omega_t
            u_t = np.random.normal(0, 0.1, 2)
            u_omega_t = np.random.normal(0, 0.1)

            # Update omega_t based on the model
            if np.linalg.norm(x_tilde[2:4]) != 0:
                omega_t = np.random.choice([-0.1, 0, 0.1]) / np.linalg.norm(x_tilde[2:4]) + u_omega_t

            # Define A and B matrices
            A = np.array([
                [1, 0, np.sin(omega_t * ts) / omega_t if omega_t != 0 else ts,
                 -(1 - np.cos(omega_t * ts)) / omega_t if omega_t != 0 else 0],
                [0, 1, -(1 - np.cos(omega_t * ts)) / omega_t if omega_t != 0 else 0,
                 np.sin(omega_t * ts) / omega_t if omega_t != 0 else ts],
                [0, 0, np.cos(omega_t * ts), -np.sin(omega_t * ts)],
                [0, 0, np.sin(omega_t * ts), np.cos(omega_t * ts)]
            ])
            B = np.array([[ts**2 / 2, 0], [0, ts**2 / 2], [ts, 0], [0, ts]])

            # Update state x_tilde
            x_tilde = A @ x_tilde + B @ u_t

            # Measurement model
            p_t = x_tilde[:2]
            h1 = 10 * np.log10(p0 / np.linalg.norm(r - p_t)**alpha)
            h2 = np.arctan2(p_t[1] - r[1], p_t[0] - r[0])
            v_t = np.random.normal(0, 0.1, 2)

            # Store the state and observation
            states[batch, t, :4] = x_tilde
            states[batch, t, 4] = omega_t
            observations[batch, t, :] = np.array([h1, h2]) + v_t

    return states, observations

# Example usage
batch_size = 5
sequence_length = 10
states, observations = generate_data(batch_size, sequence_length)

# Displaying a small sample of the generated data for verification
a, b=states[:2, :3, :], observations[:2, :3, :]  # Displaying first two batches and first three sequences
