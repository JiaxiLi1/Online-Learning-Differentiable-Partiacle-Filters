import numpy as np


def generate_reshaped_data(batch_size, sequence_length, ts=1.0, p0=1.0, alpha=2.0, r=np.array([0.0, 0.0]),
                           state_noise_mean=0.0, state_noise_scale=0.1, turn_rate_noise_mean=0.0,
                           turn_rate_noise_scale=0.1, observation_noise_mean=0.0, observation_noise_scale1=0.1, observation_noise_scale2=0.1, a_t=0):
    # Initialize the data containers
    states = np.zeros((batch_size, sequence_length, 5))  # Including omega_t
    observations = np.zeros((batch_size, sequence_length, 2))  # Two observation components

    for batch in range(batch_size):
        # Initial state
        x_tilde = np.array([0.0, 0.0, 55/np.sqrt(2), 55/np.sqrt(2)])  # Initial position and velocity
        omega_t = 0.0  # Initial turn rate

        for t in range(sequence_length):

            # Update omega_t based on the model
            if np.linalg.norm(x_tilde[2:4]) != 0:
                omega_t = a_t / np.linalg.norm(x_tilde[2:4]) + np.random.normal(turn_rate_noise_mean, turn_rate_noise_scale)

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
            x_tilde = A @ x_tilde + B @ np.random.normal(state_noise_mean, state_noise_scale, 2)

            # Measurement model
            p_t = x_tilde[:2]
            h1 = 10 * np.log10(p0 / np.linalg.norm(r - p_t)**alpha)
            h2 = np.arctan2(p_t[1] - r[1], p_t[0] - r[0])

            # Store the state and observation
            states[batch, t, :4] = x_tilde
            states[batch, t, 4] = omega_t
            observations[batch, t, :] = np.array([h1, h2]) + 0.7*np.random.normal(observation_noise_mean, observation_noise_scale1, 2) + 0.3*np.random.normal(observation_noise_mean, observation_noise_scale2, 2)

    # Reshape data into lists of numpy arrays
    state_list = [states[:, t, :] for t in range(sequence_length)]
    observation_list = [observations[:, t, :] for t in range(sequence_length)]
    # Return data as a tuple
    return (state_list, observation_list)


s_offline, o_offline = generate_reshaped_data(batch_size=500, sequence_length=51,ts=5.0, p0=1.0, alpha=2.0, r=np.array([2.0, 2.0]),
                                     state_noise_mean=0.0, state_noise_scale=0.01,
                                     turn_rate_noise_mean=0.0, turn_rate_noise_scale=0.0001,
                                     observation_noise_mean=0.0, observation_noise_scale1=0.4, observation_noise_scale2=0.25, a_t=5)

s_online, o_online = generate_reshaped_data(batch_size=1, sequence_length=5000,ts=5.0, p0=1.0, alpha=2.0, r=np.array([2.0, 2.0]),
                                     state_noise_mean=0.0, state_noise_scale=0.01,
                                     turn_rate_noise_mean=0.0, turn_rate_noise_scale=0.0001,
                                     observation_noise_mean=0.0, observation_noise_scale1=0.4, observation_noise_scale2=0.25, a_t=-5)

np.savez('offline_data.npz', states=s_offline, observations=o_offline)
np.savez('online_data.npz', states=s_online, observations=o_online)
