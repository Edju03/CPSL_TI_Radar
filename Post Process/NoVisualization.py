import numpy as np
from MMWaveDevice import MMWaveDevice
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from decimal import Decimal

timestamps_0 = []
timestamps_1 = []

def read_adc_data(adc_data_bin_file, mmwave_device_0, mmwave_device_1):
    num_samples_0 = mmwave_device_0.num_sample_per_chirp
    num_samples_1 = mmwave_device_1.num_sample_per_chirp
    num_chirps_0 = mmwave_device_0.num_chirp_per_frame 
    num_chirps_1 = mmwave_device_1.num_chirp_per_frame
    num_frames = mmwave_device_0.num_frame
    num_rx = mmwave_device_0.num_rx_chnl  
    num_lanes = 4  

    adc_data = np.fromfile(adc_data_bin_file, dtype=np.int16)
    expected_size = (num_samples_0 * (num_chirps_0) + num_chirps_1 * num_samples_1) * num_frames * num_rx * 2
    if adc_data.size != expected_size:
        raise ValueError(f"Size of the adc data ({adc_data.size}) does not match the expected size ({expected_size})")

    if mmwave_device_0.adc_bits != 16:
        l_max = 2**(mmwave_device_0.adc_bits - 1) - 1
        adc_data[adc_data > l_max] -= 2**mmwave_device_0.adc_bits

    if mmwave_device_0.is_iq_swap:
        adc_data = adc_data.reshape(-1, num_lanes).T
    else:
        adc_data = adc_data.reshape(-1, num_lanes * 2).T
        adc_data = adc_data[:num_lanes] + 1j * adc_data[num_lanes:]

    adc_data = adc_data.T.flatten()

    adc_data_0 = []
    adc_data_1 = []
    
    chirp_size_0 = num_samples_0 * num_rx 
    chirp_size_1 = num_samples_1 * num_rx 
    counter = 0
    
    for frame_idx in range(num_frames):
        for chirp_idx in range(num_chirps_0):
            # if chirp_idx % 2 == 0:
                start_idx = frame_idx * (num_chirps_0 + num_chirps_1) * (num_samples_0 * num_rx) + (chirp_idx * 2) * chirp_size_0
                end_idx = start_idx + chirp_size_0
                chirp_data = adc_data[start_idx:end_idx]
                adc_data_0.append(chirp_data)
                counter = counter + 1
            # else:
                start_idx = frame_idx * (num_chirps_0 + num_chirps_1) * (num_samples_1 * num_rx) + (chirp_idx * 2 + 1) * chirp_size_1
                end_idx = start_idx + chirp_size_1
                chirp_data = adc_data[start_idx:end_idx]
                adc_data_1.append(chirp_data)
    
    adc_data_0 = np.concatenate(adc_data_0)
    adc_data_1 = np.concatenate(adc_data_1)

    expected_shape_0 = (num_frames, num_chirps_0, num_samples_0, num_rx)
    expected_shape_1 = (num_frames, num_chirps_1, num_samples_1, num_rx)
    print(f"Expected shape for adc_data_0: {expected_shape_0}")
    print(f"Expected shape for adc_data_1: {expected_shape_1}")

    adc_data_0 = adc_data_0.reshape((num_frames, num_chirps_0, num_samples_0, num_rx)).transpose(2, 1, 0, 3)
    adc_data_1 = adc_data_1.reshape((num_frames, num_chirps_1, num_samples_1, num_rx)).transpose(2, 1, 0, 3)
    
    return adc_data_0, adc_data_1


def cfar_ca1d(magnitude, num_train, num_guard, rate_fa):
    n = len(magnitude)
    alpha = num_train * (rate_fa ** (-1 / num_train) - 1)
    peak_idx = []

    for i in range(num_guard, num_train + num_guard):
        lagging_train = magnitude[i + num_guard + 1:i + num_guard + 1 + num_train]
        noise_level = np.sum(lagging_train) / num_train
        threshold = alpha * noise_level
        if magnitude[i] > threshold:
            peak_idx.append(i)

    for i in range(num_train + num_guard, n - num_train - num_guard):
        leading_train = magnitude[i - num_train - num_guard:i - num_guard]
        lagging_train = magnitude[i + num_guard + 1:i + num_guard + 1 + num_train]
        noise_level = (np.sum(leading_train) + np.sum(lagging_train)) / (2 * num_train)
        threshold = alpha * noise_level
        if magnitude[i] > threshold:
            peak_idx.append(i)

    for i in range(n - num_train - num_guard, n - num_guard):
        leading_train = magnitude[i - num_train - num_guard:i - num_guard]
        noise_level = np.sum(leading_train) / num_train
        threshold = alpha * noise_level
        if magnitude[i] > threshold:
            peak_idx.append(i)

    return np.array(peak_idx)

def analyze_periods(timestamps):
    timestamps = np.array([Decimal(str(t[0])) for t in timestamps])
    periods = np.diff(timestamps) 

    periods_reshaped = periods.reshape(-1, 1)
    dbscan = DBSCAN(eps=0.005, min_samples=10).fit(periods_reshaped)
    labels = dbscan.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster_label = unique_labels[np.argmax(counts)]

    filtered_periods = periods[labels == main_cluster_label]
    outliers = periods[labels != main_cluster_label]

    unique_periods, counts = np.unique(filtered_periods, return_counts=True)

    return unique_periods, counts, outliers

def analyze_slopes(timestamps):
    times = np.array([Decimal(str(t[0])) for t in timestamps])
    freqs = np.array([Decimal(str(t[1])) for t in timestamps])

    print(freqs)
    print(times)
    print("Freqs:", np.diff(freqs))
    print("Times:", np.diff(times))
    
    slopes = np.diff(freqs) / np.diff(times)
    print(slopes)
    positive_slopes = slopes[slopes > 0]
    
    slopes_reshaped = positive_slopes.reshape(-1, 1)
    dbscan = DBSCAN(eps=0.01, min_samples=10).fit(slopes_reshaped)
    labels = dbscan.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster_label = unique_labels[np.argmax(counts)]

    filtered_slopes = positive_slopes[labels == main_cluster_label]
    outliers = positive_slopes[labels != main_cluster_label]

    unique_slopes, counts = np.unique(filtered_slopes, return_counts=True)

    return unique_slopes, counts, outliers

def analyze_interference_batches(timestamps):
    batches = []
    batch_start_indices = [0]
    
    for i in range(1, len(timestamps)):
        if Decimal(str(timestamps[i][0])) - Decimal(str(timestamps[i-1][0])) > Decimal('50'):
            batch_start_indices.append(i)

    batch_start_times = [Decimal(str(timestamps[i][0])) for i in batch_start_indices]
    batch_periods = np.diff(batch_start_times)

    if len(batch_periods) == 0:
        return [], [], []

    batch_periods_reshaped = batch_periods.reshape(-1, 1)
    dbscan = DBSCAN(eps=0.1, min_samples=5).fit(batch_periods_reshaped)
    labels = dbscan.labels_

    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster_label = unique_labels[np.argmax(counts)]

    filtered_batch_periods = batch_periods[labels == main_cluster_label]
    outliers = batch_periods[labels != main_cluster_label]

    unique_batch_periods, counts = np.unique(filtered_batch_periods, return_counts=True)

    return unique_batch_periods, counts, outliers

def process_adc_data(adc_data, num_samples, num_chirps, num_frames, rx_channel, sample_rate, frame_periodicity, idle_time, ramp_time, adc_start_time, mmwave_devices):
    global timestamps_0, timestamps_1

    for frame_idx in range(num_frames):
        print("Processing Frame:", frame_idx)
        for chirp_idx in range(num_chirps[0]):
            chirp_data_0 = adc_data[0][:, chirp_idx, frame_idx, rx_channel]
            chirp_data_1 = adc_data[1][:, chirp_idx, frame_idx, rx_channel]

            time_per_sample = 1 / sample_rate[0] 
            start_time_0 = frame_idx * frame_periodicity[0] + (2 * chirp_idx) * (idle_time[0] + ramp_time[0]) + (idle_time[0] + adc_start_time[0])
            start_time_1 = frame_idx * frame_periodicity[1] + (2 * chirp_idx + 1) * (idle_time[1] + ramp_time[1]) + (idle_time[1] + adc_start_time[1])
            time_indices_0 = start_time_0 + np.arange(num_samples[0]) * time_per_sample
            time_indices_1 = start_time_1 + np.arange(num_samples[1]) * time_per_sample

            # print("First Used:", start_time_0, start_time_1)
            # print("Last Used:", time_indices_0[-1], time_indices_1[-1])
            # print(idle_time[0] + ramp_time[0], idle_time[1] + ramp_time[1])
            # print(frame_periodicity[0], frame_periodicity[1])

            complex_magnitude_0 = np.abs(chirp_data_0)
            complex_magnitude_1 = np.abs(chirp_data_1)
            
            num_train = 10  
            num_guard = 2   
            rate_fa = 1e-2  
            
            peak_idx_0 = cfar_ca1d(complex_magnitude_0, num_train, num_guard, rate_fa)
            peak_idx_1 = cfar_ca1d(complex_magnitude_1, num_train, num_guard, rate_fa)

            for idx in peak_idx_0:
                if not any(np.isclose(time_indices_0[idx], t[0]) for t in timestamps_0) and (len(timestamps_0) == 0 or time_indices_0[idx] > timestamps_0[-1][0] + 0.001):
                    timestamps_0.append((time_indices_0[idx], mmwave_devices[0].freq / 1e9))

            for idx in peak_idx_1:
                if not any(np.isclose(time_indices_1[idx], t[0]) for t in timestamps_1) and (len(timestamps_1) == 0 or time_indices_1[idx] > timestamps_1[-1][0] + 0.001):
                    timestamps_1.append((time_indices_1[idx], mmwave_devices[1].freq / 1e9))

def main():
    adc_data_bin_file = '/Users/edwardju/Downloads/adc_data_LowSlopeTest3.bin'
    mmwave_setup_json_file = '/Users/edwardju/Downloads/LowSlopeTest2.mmwave.json'

    mmwave_device_profile_0 = MMWaveDevice(adc_data_bin_file, mmwave_setup_json_file, profile_id=0)
    mmwave_device_profile_0.print_device_configuration()

    mmwave_device_profile_1 = MMWaveDevice(adc_data_bin_file, mmwave_setup_json_file, profile_id=1)
    mmwave_device_profile_1.print_device_configuration()

    adc_data_0, adc_data_1 = read_adc_data(adc_data_bin_file, mmwave_device_profile_0, mmwave_device_profile_1)

    num_samples = [mmwave_device_profile_0.num_sample_per_chirp, mmwave_device_profile_1.num_sample_per_chirp]
    num_chirps = [mmwave_device_profile_0.num_chirp_per_frame, mmwave_device_profile_1.num_chirp_per_frame]
    num_frames = mmwave_device_profile_0.num_frame
    rx_channel = 0

    sample_rate = [mmwave_device_profile_0.adc_samp_rate * 1000, mmwave_device_profile_1.adc_samp_rate * 1000]
    frame_periodicity = [mmwave_device_profile_0.frame_periodicity, mmwave_device_profile_1.frame_periodicity]
    idle_time = [mmwave_device_profile_0.chirp_idle_time * 1e-3, mmwave_device_profile_1.chirp_idle_time * 1e-3]
    ramp_time = [mmwave_device_profile_0.chirp_ramp_time * 1e-3, mmwave_device_profile_1.chirp_ramp_time * 1e-3]
    adc_start_time = [mmwave_device_profile_0.chirp_adc_start_time * 1e-3, mmwave_device_profile_1.chirp_adc_start_time * 1e-3]

    process_adc_data([adc_data_0, adc_data_1], num_samples, num_chirps, num_frames, rx_channel, sample_rate, frame_periodicity, idle_time, ramp_time, adc_start_time, [mmwave_device_profile_0, mmwave_device_profile_1])

    if len(timestamps_0) > 1:
        # timestamps_0.sort(key=lambda x: x[0])
        unique_periods_0, counts_0, outliers_0 = analyze_periods(timestamps_0)
        unique_batch_periods_0, batch_counts_0, batch_outliers_0 = analyze_interference_batches(timestamps_0)
        # print("Profile 0 - Timestamps:", timestamps_0)
        # print("Profile 0 - Unique Periods:", unique_periods_0, "Counts:", counts_0)
        # print("Profile 0 - Batch Periods:", unique_batch_periods_0, "Counts:", batch_counts_0)

    if len(timestamps_1) > 1:
        # timestamps_1.sort(key=lambda x: x[0])
        unique_periods_1, counts_1, outliers_1 = analyze_periods(timestamps_1)
        unique_batch_periods_1, batch_counts_1, batch_outliers_1 = analyze_interference_batches(timestamps_1)
        # print("Profile 1 - Timestamps:", timestamps_1)
        # print("Profile 1 - Unique Periods:", unique_periods_1, "Counts:", counts_1)
        # print("Profile 1 - Batch Periods:", unique_batch_periods_1, "Counts:", batch_counts_1)

        timestamps = timestamps_0 + timestamps_1
        timestamps.sort(key=lambda x: x[0])
        unique_slopes, slope_counts, slope_outliers = analyze_slopes(timestamps)
        print(slope_outliers)
        print("Slopes:", unique_slopes, "Counts:", slope_counts)

    print("\nEstimated Results: \n")

    average_slope = np.average(unique_slopes, weights=slope_counts)
    print("Weighted Average Slope:", average_slope)

    average_period_0 = np.average(unique_periods_0, weights=counts_0)
    print("Weighted Average Period (Profile 0):", average_period_0)

    average_period_1 = np.average(unique_periods_1, weights=counts_1)
    print("Weighted Average Period (Profile 1):", average_period_1)

    average_batch_period_0 = np.average(unique_batch_periods_0, weights=batch_counts_0)
    print("Weighted Average Batch Period (Profile 0):", average_batch_period_0)

    average_batch_period_1 = np.average(unique_batch_periods_1, weights=batch_counts_1)
    print("Weighted Average Batch Period (Profile 1):", average_batch_period_1)

if __name__ == "__main__":
    main()
