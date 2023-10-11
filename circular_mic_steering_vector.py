import numpy as np
import matplotlib.pyplot as plt


# ステアリングベクトルを算出: Far-Field仮定
# mic_position: 3 x M  dimensional ndarray [[x,y,z],[x,y,z]]
# source_position: 3x Ns dimensional ndarray [[x,y,z],[x,y,z] ]
# freqs: Nk dimensional array [f1,f2,f3...]
# sound_speed: 音速 [m/s]
# return: steering vector (Nk x Ns x M)
def calculate_steering_vector(mic_alignments, source_locations, freqs, sound_speed=340):
    # マイク数
    n_channels = np.shape(mic_alignments)[1]
    # 音源位置を正規化
    # 音源位置をL2ノルムで割るので、3次元の方向ベクトルになる
    norm_source_locations = source_locations / np.linalg.norm(source_locations, 2, axis=0, keepdims=True)
    # 位相を求める
    # k,ism,ism->ksmでi方向の和を取る
    steering_phase = np.einsum("k,ism,ism->ksm", 2.0j * np.pi / sound_speed * freqs,
                               norm_source_locations[..., None],
                               mic_alignments[:, None, :])
    # print(f"{(2.0j * np.pi / sound_speed * freqs).shape=}")
    # print(f"{norm_source_locations[..., None].shape=}")
    # print(f"{mic_alignments[:, None, :].shape=}")
    # print(f"{steering_phase.shape=}")
    # ステアリングベクトルを算出
    steering_vector = 1.0 / np.sqrt(n_channels) * np.exp(steering_phase)

    return steering_vector


# Number of microphones
num_mics = 8

# Radius of the circle on which the microphones are placed
radius = 0.1

# Angular positions of the microphones
angles = np.linspace(0, 2*np.pi, num_mics, endpoint=False)

# Microphone positions
mic_x = radius * np.cos(angles)
mic_y = radius * np.sin(angles)
mic_z = np.zeros_like(mic_x)

# Combining x, y, z coordinates
mic_alignments = np.vstack([mic_x, mic_y, mic_z])

# # Visualization of Microphone Positions
# plt.figure()
# plt.scatter(mic_x, mic_y, label="Microphones", marker="o")
# plt.title("Microphone Positions")
# plt.xlabel("X [m]")
# plt.ylabel("Y [m]")
# plt.axis("equal")
# plt.grid(True)
# plt.legend()
# plt.show()

# Using the same source locations and frequency settings from the original code
sample_rate = 16000
N = 1024
Nk = N / 2 + 1
freqs = np.arange(0, Nk, 1) * sample_rate / N
doas = np.array([[np.pi / 2, 0], [np.pi / 2, np.pi]])
distance = 3.01
source_locations = np.zeros((3, doas.shape[0]), dtype=doas.dtype)
source_locations[0, :] = np.cos(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[1, :] = np.sin(doas[:, 1]) * np.sin(doas[:, 0])
source_locations[2, :] = np.cos(doas[:, 0])
source_locations *= distance

# Calculate steering vectors
steering_vectors = calculate_steering_vector(mic_alignments, source_locations, freqs)

# Print shape and inner product
print(f"{steering_vectors.shape=}")
