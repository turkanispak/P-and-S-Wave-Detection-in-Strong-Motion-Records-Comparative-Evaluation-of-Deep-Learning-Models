from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def modified_adaptive_window(test_vector, pass_type, elimination_type, N, stride):
    # Parameters
    flag = 0  # Raise if adaptive window is executed

    # Calculate the remainder after evenly dividing the length of test_vector
    remainder = len(test_vector) % N

    # Initialize valid_frames with the entire test_vector
    num_frames = int(np.ceil(len(test_vector) / N))

    # Calculate dimensions for splitting
    if remainder == 0:
        dimensions = [N] * num_frames
    else:
        dimensions = [N] * (num_frames - 1) + [remainder]

    # Check if the sum of dimensions matches the size of test_vector
    if sum(dimensions) != len(test_vector):
        print('Error: Sum of dimensions does not match the size of test_vector.')
    else:
        # Split test_vector into valid_frames
        valid_frames = [test_vector[sum(dimensions[:i]):sum(dimensions[:i+1])] for i in range(num_frames)]

    # Calculate the average of test_vector
    test_vector_avg = np.mean(np.abs(test_vector))

    # Define traversal indices based on pass_type
    if pass_type == 'forward':
        start_index = 0
        end_index = len(valid_frames)
        step = 1
    elif pass_type == 'backward':
        start_index = len(valid_frames) - 1
        end_index = -1
        step = -1
    else:
        raise ValueError("Invalid pass_type. Pass_type must be either 'forward' or 'backward'.")

    # Traverse the frames
    condition = False
    for i in range(start_index, end_index, step):
        frame_avg = np.mean(np.abs(valid_frames[i]))

        # Eliminate frames based on elimination_type
        if elimination_type == 'greater_than':
            condition = test_vector_avg > frame_avg
        elif elimination_type == 'less_than':
            condition = test_vector_avg <= frame_avg
        else:
            raise ValueError("Invalid elimination_type. Elimination_type must be either 'greater_than' or 'less_than'.")

        # If not met, remove the frame
        valid_frames.pop(i)

        # Break the loop if condition met
        if condition:
            break
        
        # Update the index based on the stride
        i += stride

    # If adaptive window did not detect anything, remove last frame as buffer
    if not condition:
        valid_frames.pop(i)

    # Combine valid frames into a single vector
    result = np.concatenate(valid_frames)
    maw_index = i * len(valid_frames)

    return result, maw_index

def inject_simulated_noise(noise_data, simulated_noise, injection_index):
    # Check if injection_index is within bounds
    if injection_index < 1 or injection_index > len(noise_data):
        raise ValueError('Injection index is out of bounds.')
    combined_noise = np.concatenate((noise_data[:injection_index-1], simulated_noise, noise_data[injection_index-1:]))
    return combined_noise

def find_insertion_index(a):
    noise_data = np.abs(a)
    window_size = 5
    num_samples = len(noise_data)
    forward_index = 0
    backward_index = num_samples - 1
    forward_characteristic = []
    backward_characteristic = []
    
    while forward_index < backward_index:
        forward_window = noise_data[forward_index:min(forward_index + window_size, num_samples)]
        backward_window = noise_data[max(backward_index - window_size + 1, 0):backward_index + 1]
        
        if len(forward_window) == 0 or len(backward_window) == 0:
            print("Unable to find characteristics.")
            insertion_index = 1
            return insertion_index
        
        forward_mean = np.mean(forward_window)
        backward_mean = np.mean(backward_window)
        print("Means:")
        print(forward_mean)
        print(backward_mean)
        
        if forward_mean <= 0.5 * backward_mean:
            forward_characteristic.extend(forward_window)
            backward_characteristic = backward_window.tolist() + backward_characteristic
        else:
            print("Break")
            break
        
        forward_index += window_size
        backward_index -= window_size
    
    print("Done")
    print(len(forward_characteristic))
    insertion_index = len(forward_characteristic)
    return insertion_index

def fractional_brownian_motion(extracted_noise_channel2, expected_output_size):
    # Load accelerometer noise data
    noise_data = extracted_noise_channel2
    
    # Calculate autocorrelation function
    autocorr_noise = np.correlate(noise_data, noise_data, mode='full')
    autocorr_noise = autocorr_noise / np.max(autocorr_noise)  # Normalize
    
    # Find the first zero crossing to estimate the dominant period of the noise
    zero_crossings = np.where(autocorr_noise <= 0)[0]
    dominant_period = np.mean(np.diff(zero_crossings))
        
    # Define range of parameters to search
    H_range = np.arange(0.1, 2.1, 0.1)  # Range of Hurst exponent values to try
    f_low = 1 / dominant_period  # Lower corner frequency based on dominant period (Hz)
    coeff_fh = 0.65
    f_high = 1 / (coeff_fh * dominant_period)  # Higher corner frequency as multiple of dominant period (Hz)
    
    # Initialize variables to store the best parameters and minimum difference
    best_H = np.nan
    min_difference = np.inf
    best_simulated_noise = []  # Initialize variable to store the best simulated noise
    
    # Loop through Hurst exponent values
    for H in H_range:
        # Generate amplitude-based noise using the Brownian-motion-based parametric modeling algorithm
        N = expected_output_size  # Length of the output noise data
        B_H = fractal_noise(N, H)  # Generate FBM series
        h = bandpass_filter(N, f_low, f_high)  # Generate band-pass filter
        simulated_noise = np.convolve(B_H, h, mode='same')  # Simulate noise
        
        # Normalize the simulated noise to have the same mean and standard deviation as the original noise data
        simulated_noise = simulated_noise - np.mean(simulated_noise)  # Remove DC offset
        simulated_noise = simulated_noise * np.std(noise_data) / np.std(simulated_noise)  # Scale to match standard deviation
        
        # Calculate mean squared error between original and simulated noise
        difference = np.mean((noise_data - simulated_noise) ** 2)
        
        # Update best parameters if the difference is minimized
        if difference < min_difference:
            best_H = H
            min_difference = difference
            best_simulated_noise = simulated_noise[:expected_output_size]  # Store the best simulated noise
    
    best_simulated_result = np.flip(best_simulated_noise)
    return best_simulated_result

def fractal_noise(N, H):
    # Initialization
    B_H = np.zeros(N)
    # RMD algorithm
    for i in range(1, N-1):
        B_H[i] = 0.5 * (B_H[i-1] + B_H[i+1]) + np.random.randn() / np.sqrt(2 ** (2 * H))
    return B_H

def bandpass_filter(N, f_low, f_high):
    # Frequency vector
    f = np.linspace(0, 1, N)
    # Ideal band-pass filter response
    H_bp = (f > f_low) & (f < f_high)
    # Inverse Fourier transform to get time-domain filter
    h = np.real(np.fft.ifft(np.fft.fftshift(H_bp)))
    return h

# ---- Helpers ----
def tvec(n, dt):
    """1-based time vector: dt, 2*dt, ..., n*dt."""
    return np.arange(1, n + 1, dtype=float) * dt

def _closest_index(x, value):
    """0-based index of element in x closest to value."""
    return int(np.argmin(np.abs(x - value)))

# === MAIN BLOCK ===

# ---- Colors / layout  ----
colorChannel1 = np.array([68, 68, 68]) / 255.0
colorChannel2 = np.array([0, 0, 0], dtype=float)
colorChannel3 = np.array([136, 136, 136]) / 255.0

colorPTime = np.array([0.8500, 0.3250, 0.0980])
colorSTime = np.array([0.0, 0.4470, 0.7410])
horizontal_res = 1920
vertical_res = 1080
subplotRows = 5
subplotCols = 1

# ---- Input folder (script dir / "Sample Data") ----
script_dir = Path(__file__).resolve().parent
folderPath = script_dir / "Sample Data"  # Change this to your data folder

# ---- Iterate over *.mat files ----
fileList = sorted(folderPath.glob("*.mat"))
for i, mat_path in enumerate(fileList, start=1):
    print(f"Processing file: {mat_path.name}")
    fileName = mat_path.name
    try:
        # Load the .mat (R2022a-compatible; squeeze for struct-like access)
        data = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)

        # Access nested fields like MATLAB: data.EQ.anEQ.Accel
        EQ = data["EQ"]
        anEQ = EQ.anEQ
        accel_data_orig = np.asarray(anEQ.Accel)
        accel_data = accel_data_orig.copy() # Copy to avoid modifying original data

        # Time base
        time_interval = 0.01
        time_values = tvec(accel_data.shape[0], time_interval)

        # P/S times and site info
        actualPTime = float(anEQ.Ptime)
        actualSTime = float(anEQ.Stime)
        epicenter = np.asarray(anEQ.epicenter).ravel()
        statco = np.asarray(anEQ.statco).ravel()
        lat1, lon1 = epicenter[0], epicenter[1]
        lat2, lon2 = statco[0], statco[1]

        # Indices closest to P/S times
        indexPTime = _closest_index(time_values, actualPTime)   
        indexSTime = _closest_index(time_values, actualSTime)   

        # ================= Extraction of the noise signal =================
        k = accel_data[:indexPTime + 1, 1].astype(float)

        # Combine valid frames (exact call)
        extracted_noise_channel2, _ = modified_adaptive_window(
            k, 'backward', 'greater_than', 60, 0
        )

        # Remove the identified "artifact" prefix from the extracted noise
        injection_index_art = find_insertion_index(extracted_noise_channel2)  # returns 1-based
        extracted_artefact_CH2 = extracted_noise_channel2[:injection_index_art].copy()

        # Call the noise generation function on the remainder to simulate noise
        extracted_noise_channel2 = extracted_noise_channel2[injection_index_art:].copy()
        generated_noise_size = len(extracted_noise_channel2)
        simulated_noise = fractional_brownian_motion(
            extracted_noise_channel2, generated_noise_size
        )

        # Inject generated noise to extracted noise
        injection_index = find_insertion_index(extracted_noise_channel2)  # 1-based
        if injection_index == 0:
            injection_index = 1

        combined_brownian_noise = inject_simulated_noise(
            extracted_noise_channel2, simulated_noise, injection_index
        )

        # Place the combined noise to channel 2 data
        ch2 = accel_data[:, 1].astype(float)
        tail_start_1based = injection_index + len(simulated_noise) + len(extracted_artefact_CH2) + 1
        # Convert that 1-based start to 0-based slice index
        tail_start_0based = max(0, tail_start_1based - 1)
        tail_start_0based = min(tail_start_0based, len(ch2))  # clamp
        result_signal = np.concatenate([combined_brownian_noise, ch2[tail_start_0based:]])

        # Update time_values for result length
        time_values = tvec(len(result_signal), time_interval)

        # ========================= ONE-FIGURE SUMMARY PLOT =========================
        # Freeze originals and derive plot-specific vectors
        N_orig = accel_data.shape[0]
        t_orig = tvec(N_orig, time_interval)

        t_noise_extracted = tvec(len(extracted_noise_channel2), time_interval)
        t_noise_artifact = tvec(len(extracted_artefact_CH2), time_interval)
        t_noise_sim = tvec(len(simulated_noise), time_interval)

        N_final = len(result_signal)
        t_final = tvec(N_final, time_interval)

        # Adjusted P/S times after removal/addition of sample points
        actualPTime_adjusted = actualPTime - len(extracted_artefact_CH2) * time_interval + len(simulated_noise) * time_interval
        actualSTime_adjusted = actualSTime - len(extracted_artefact_CH2) * time_interval + len(simulated_noise) * time_interval

        # ---- Figure + layout ----
        fig = plt.figure(figsize=(horizontal_res / 100.0, vertical_res / 100.0), dpi=100, facecolor="w")
        fig.suptitle(f"Augmentation Summary — {fileName}", fontsize=14)

        gs = fig.add_gridspec(subplotRows, subplotCols, hspace=0.65)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[3, 0])
        ax5 = fig.add_subplot(gs[4, 0]) if subplotRows == 5 else None  # keep 5th if you want

        # (1) Original
        ax = ax1
        ax.plot(t_orig, accel_data[:, 1], color=colorChannel2, linewidth=0.8)
        sharedY = ax.get_ylim()
        ax.axvline(actualPTime, linestyle="--", color=colorPTime, linewidth=1.5, label="P")
        ax.axvline(actualSTime, linestyle="--", color=colorSTime, linewidth=1.5, label="S")
        ax.set_title("Original Earthquake Accelerogram Record")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration")
        ax.set_xlim(t_orig[0], t_orig[-1]); ax.set_ylim(sharedY)

        # (2) Noise & Artifact Detected
        ax = ax2
        ax.plot(t_orig, accel_data[:, 1], color=colorChannel2, linewidth=0.8, label="Accelerometer Data")
        ax.axvline(actualPTime, linestyle="--", color=colorPTime, linewidth=1.5, label="P-time")
        ax.axvline(actualSTime, linestyle="--", color=colorSTime, linewidth=1.5, label="S-time")
        if len(extracted_noise_channel2) > 0:
            ax.plot(
                (len(extracted_artefact_CH2) + np.arange(1, len(extracted_noise_channel2) + 1)) * time_interval,
                extracted_noise_channel2,
                "r",
                linewidth=2,
                label="Noise",
            )
        if len(extracted_artefact_CH2) > 0:
            ax.plot(t_noise_artifact, extracted_artefact_CH2, "g", linewidth=1.5, label="Noise Artifact")
        ax.set_title("Noise and Artifact Detected Earthquake Accelerogram Record")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration")
        ax.set_xlim(t_orig[0], t_orig[-1]); ax.set_ylim(sharedY)
        ax.legend(loc="upper right")

        # (3) Detected Noise
        ax = ax3
        if len(extracted_noise_channel2) > 0:
            ax.plot(t_noise_extracted, extracted_noise_channel2, "r", linewidth=2, label="Detected Noise")
            ax.set_xlim(t_noise_extracted[0], t_noise_extracted[-1])
        else:
            ax.plot([np.nan], [np.nan], "r", label="Detected Noise")
        ax.set_title("Detected Noise")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration")
        ax.legend(loc="upper right")

        # (4) Simulated Fractional Brownian Motion Noise
        ax = ax4
        if len(simulated_noise) > 0:
            ax.plot(t_noise_sim, simulated_noise, "b", linewidth=1.5, label="Fractional Brownian Motion Noise")
            ax.set_xlim(t_noise_sim[0], t_noise_sim[-1])
        else:
            ax.plot([np.nan], [np.nan], "b", label="Fractional Brownian Motion Noise")
        ax.set_title("Simulated Fractional Brownian Motion Noise")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration")
        ax.legend(loc="upper right")

        # (5) Final Generated Signal
        if subplotRows == 5 and ax5 is not None:
            ax = ax5
            ax.plot(t_final, result_signal, color=colorChannel2, linewidth=1.0, label="Channel 2 (Final)")

            # Combined FBM noise at the front
            if len(combined_brownian_noise) > 0:
                L_cbn = len(combined_brownian_noise)
                ax.plot(t_final[:L_cbn], combined_brownian_noise, "r", linewidth=1.2, label="Combined FBM Noise")

            # Overlay generated FBM at its injection region on the final signal
            idx_start = max(0, injection_index - 1) 
            idx_end = min(idx_start + len(simulated_noise), len(t_final))
            if idx_end > idx_start and len(simulated_noise) > 0:
                ax.plot(t_final[idx_start:idx_end], simulated_noise[:(idx_end - idx_start)],
                        "b", linewidth=1.2, label="Generated FBM Noise")

            # Adjusted P/S markers
            ax.axvline(actualPTime_adjusted, linestyle="--", color=colorPTime, linewidth=1.2, label="P-time (adj.)")
            ax.axvline(actualSTime_adjusted, linestyle="--", color=colorSTime, linewidth=1.2, label="S-time (adj.)")
            ax.set_title("Final Generated Signal")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration")
            ax.set_xlim(t_final[0], t_final[-1])
            ax.legend(loc="upper right")

        # Save and close
        out_dir = script_dir / "Figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"{mat_path.stem}_augmentation_summary.png"
        fig.savefig(out_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  + Saved summary figure -> {out_png}")

    except Exception as exception:
        print(f"Error processing file {fileName}:\n{exception}")
