import timeit

# pythonファイルからプログラムを読み込む
with open("einsum_valid.py", "r") as f:
    code = f.read()

# Define the code for the functions to be timed
einsum_code = "calculate_steering_vector_einsum(mic_alignments, source_locations, freqs)"
no_einsum_code = "calculate_steering_vector_no_einsum(mic_alignments, source_locations, freqs)"

# Time the execution of both methods
time_einsum = timeit.timeit(einsum_code, setup=code, number=1000)
time_no_einsum = timeit.timeit(no_einsum_code, setup=code, number=1000)

# Print the results
print(f"Time for einsum: {time_einsum}")
print(f"Time for no einsum: {time_no_einsum}")

# Time for einsum: 0.48610550000012154
# Time for no einsum: 10.491286799995578
# einsumを使った方が速い