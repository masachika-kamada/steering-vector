import numpy as np

# Given function to calculate steering vector using np.einsum
def calculate_steering_vector_einsum(mic_alignments, source_locations, freqs, sound_speed=340):
    n_channels = np.shape(mic_alignments)[1]
    norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
    steering_phase = np.einsum("k,ism,ism->ksm", 2.0j * np.pi / sound_speed * freqs, norm_source_locations[..., None], mic_alignments[:, None, :])
    steering_vector = 1.0 / np.sqrt(n_channels) * np.exp(steering_phase)
    return steering_vector

# Function to calculate steering vector without using np.einsum
def calculate_steering_vector_no_einsum(mic_alignments, source_locations, freqs, sound_speed=340):
    steering_phase = np.zeros((len(freqs), source_locations.shape[1], mic_alignments.shape[1]), dtype=complex)
    phase_factor = 2.0j * np.pi / sound_speed * freqs
    # print(f"{phase_factor.shape=}")
    for k, pf in enumerate(phase_factor):
        for s in range(source_locations.shape[1]):
            for m in range(mic_alignments.shape[1]):
                # print(f"{k=}, {s=}, {m=}")
                # print(f"{pf.shape=}")
                # print(f"{norm_source_locations[:, s].shape=}")
                # print(f"{mic_alignments[:, m].shape=}")
                # print(pf * np.dot(norm_source_locations[:, s], mic_alignments[:, m]))
                steering_phase[k, s, m] = pf * np.dot(norm_source_locations[:, s], mic_alignments[:, m])
    n_channels = np.shape(mic_alignments)[1]
    steering_vector = 1.0 / np.sqrt(n_channels) * np.exp(steering_phase)
    return steering_vector

# Parameters
num_mics = 8
radius = 0.1
angles = np.linspace(0, 2*np.pi, num_mics, endpoint=False)
mic_x = radius * np.cos(angles)
mic_y = radius * np.sin(angles)
mic_z = np.zeros_like(mic_x)
mic_alignments = np.vstack([mic_x, mic_y, mic_z])
sample_rate = 16000
N = 1024
Nk = int(N / 2 + 1)
freqs = np.arange(0, Nk, 1) * sample_rate / N
doas = np.array([[np.pi / 2, 0], [np.pi / 2, np.pi]])
distance = 3.01
source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance
norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)

# Calculate steering vectors using both methods
steering_vectors_einsum = calculate_steering_vector_einsum(mic_alignments, source_locations, freqs)
steering_vectors_no_einsum = calculate_steering_vector_no_einsum(mic_alignments, source_locations, freqs)

# Check if both methods yield the same result
are_equal = np.allclose(steering_vectors_einsum, steering_vectors_no_einsum)
print(are_equal)
