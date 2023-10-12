import numpy as np


def calculate_steering_vector(mic_alignments, freqs, angle_step=1, sound_speed=340):
    n_channels = np.shape(mic_alignments)[1]
    theta = np.deg2rad(np.arange(-180, 181, angle_step))  # 0 to 359 degrees in radians
    phi = np.zeros_like(theta)  # Assuming all sources lie in the XY plane
    print(phi)
    print(freqs)

    source_locations = np.zeros((3, len(theta)), dtype=float)
    source_locations[0, :] = np.cos(phi) * np.sin(theta)
    source_locations[1, :] = np.sin(phi) * np.sin(theta)
    source_locations[2, :] = np.cos(theta)

    norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
    steering_phase = np.einsum("k,ism,ism->ksm", 2.0j * np.pi / sound_speed * freqs,
                               norm_source_locations[..., None],
                               mic_alignments[:, None, :])
    steering_vector = 1.0 / np.sqrt(n_channels) * np.exp(steering_phase)
    return steering_vector


num_mics = 8
radius = 0.1
angles = np.linspace(0, 2*np.pi, num_mics, endpoint=False)
# Microphone positions
mic_x = radius * np.cos(angles)
mic_y = radius * np.sin(angles)
mic_z = np.zeros_like(mic_x)
# Combining x, y, z coordinates
mic_alignments = np.vstack([mic_x, mic_y, mic_z])

sample_rate = 16000
N = 1024
Nk = N / 2 + 1
freqs = np.arange(0, Nk, 1) * sample_rate / N

# Calculate steering vectors for 1-degree intervals
steering_vectors = calculate_steering_vector(mic_alignments, freqs, angle_step=1)

print(f"{steering_vectors.shape=}")
