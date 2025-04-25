import streamlit as st
import numpy as np

# ------------------------- Scheduling Algorithms -------------------------

def calculate_makespan(sequence, processing_times):
    num_jobs = len(sequence)
    num_machines = len(processing_times[0])
    completion = np.zeros((num_jobs, num_machines), dtype=int)
    for i in range(num_jobs):
        job = sequence[i]
        for j in range(num_machines):
            if i == 0 and j == 0:
                completion[i][j] = processing_times[job][j]
            elif i == 0:
                completion[i][j] = completion[i][j - 1] + processing_times[job][j]
            elif j == 0:
                completion[i][j] = completion[i - 1][j] + processing_times[job][j]
            else:
                completion[i][j] = max(completion[i - 1][j], completion[i][j - 1]) + processing_times[job][j]
    return completion[-1][-1]

def johnsons_algorithm_2_machines(jobs, machine1, machine2):
    job_data = list(zip(jobs, machine1, machine2))
    list1 = [job for job in job_data if job[1] <= job[2]]
    list2 = [job for job in job_data if job[1] > job[2]]
    list1.sort(key=lambda x: x[1])
    list2.sort(key=lambda x: x[2], reverse=True)
    sequence = [job[0] for job in list1 + list2]
    return sequence

def johnsons_algorithm_3_machines(jobs, machine1, machine2, machine3):
    if min(machine1) >= max(machine2) or min(machine3) >= max(machine2):
        machine_g = [machine1[i] + machine2[i] for i in range(len(jobs))]
        machine_h = [machine2[i] + machine3[i] for i in range(len(jobs))]
        return johnsons_algorithm_2_machines(jobs, machine_g, machine_h)
    else:
        return None

def palmer_algorithm(processing_times):
    num_jobs, num_machines = len(processing_times), len(processing_times[0])
    weights = [-(num_machines - 1 - 2 * i) for i in range(num_machines)]
    slope_indices = [sum(p * w for p, w in zip(job, weights)) for job in processing_times]
    job_order = sorted(range(num_jobs), key=lambda j: slope_indices[j], reverse=True)
    return job_order

def neh(processing_times):
    num_jobs = len(processing_times)
    job_total_times = [(i, sum(processing_times[i])) for i in range(num_jobs)]
    sorted_jobs = sorted(job_total_times, key=lambda x: x[1], reverse=True)
    sequence = [sorted_jobs[0][0]]
    for i in range(1, num_jobs):
        current_job = sorted_jobs[i][0]
        best_sequence = []
        min_makespan = float('inf')
        for j in range(i + 1):
            temp_sequence = sequence[:j] + [current_job] + sequence[j:]
            makespan = calculate_makespan(temp_sequence, processing_times)
            if makespan < min_makespan:
                min_makespan = makespan
                best_sequence = temp_sequence
        sequence = best_sequence
    return sequence

def schedule_jobs(jobs_subset, m):
    jobs_array = np.array(jobs_subset)
    if m == 2:
        return johnsons_algorithm_2_machines(range(len(jobs_subset)), jobs_array[:, 0], jobs_array[:, 1])
    elif m == 3:
        johnson_seq = johnsons_algorithm_3_machines(range(len(jobs_subset)), jobs_array[:, 0], jobs_array[:, 1], jobs_array[:, 2])
        neh_seq = neh(jobs_array.tolist())
        johnson_makespan = calculate_makespan(johnson_seq, jobs_array) if johnson_seq else float('inf')
        neh_makespan = calculate_makespan(neh_seq, jobs_array)
        return johnson_seq if johnson_makespan <= neh_makespan else neh_seq
    else:
        palmer_seq = palmer_algorithm(jobs_array.tolist())
        neh_seq = neh(jobs_array.tolist())
        palmer_makespan = calculate_makespan(palmer_seq, jobs_array)
        neh_makespan = calculate_makespan(neh_seq, jobs_array)
        return palmer_seq if palmer_makespan <= neh_makespan else neh_seq

def calculate_completion_times(jobs, sequence, m):
    n = len(sequence)
    completion_times = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            job_idx = sequence[i]
            if i == 0 and j == 0:
                completion_times[i][j] = jobs[job_idx][j]
            elif i == 0:
                completion_times[i][j] = completion_times[i][j - 1] + jobs[job_idx][j]
            elif j == 0:
                completion_times[i][j] = completion_times[i - 1][j] + jobs[job_idx][j]
            else:
                completion_times[i][j] = max(completion_times[i - 1][j], completion_times[i][j - 1]) + jobs[job_idx][j]
    return completion_times, completion_times[-1][-1]

# ------------------------- Streamlit UI -------------------------

def main():
    st.title("Dynamic Job Shop Scheduling")

    m = st.number_input("Enter number of machines (2 or more):", min_value=2, step=1)
    n = st.number_input("Enter number of initial jobs:", min_value=1, step=1)

    st.write("Enter processing times for each job (space-separated for each machine):")
    jobs = []
    for i in range(n):
        job_input = st.text_input(f"Job {i+1} times:", key=f"job_{i}")
        times = list(map(int, job_input.strip().split())) if job_input else [0]*m
        times += [0] * (m - len(times))
        jobs.append(times[:m])

    if st.button("Schedule Jobs"):
        sequence = schedule_jobs(jobs, m)
        completion_times, makespan = calculate_completion_times(jobs, sequence, m)
        st.subheader("Initial Job Sequence")
        st.write([f"J{i+1}" for i in sequence])
        st.write("Makespan:", makespan)

        st.subheader("Completion Times")
        st.table(completion_times)

    if st.checkbox("Add New Jobs"):
        completed_job_idx = st.number_input("Index (1-based) of last COMPLETED job:", min_value=1, max_value=n, step=1)
        new_job_count = st.number_input("Number of new jobs to add:", min_value=1, step=1)
        new_jobs = []
        for i in range(new_job_count):
            new_input = st.text_input(f"New Job {i+1} times:", key=f"new_job_{i}")
            times = list(map(int, new_input.strip().split())) if new_input else [0]*m
            times += [0] * (m - len(times))
            new_jobs.append(times[:m])

        if st.button("Reschedule with New Jobs"):
            all_jobs = jobs + new_jobs
            old_seq = schedule_jobs(jobs, m)
            remaining = [idx for idx in old_seq if idx >= (completed_job_idx - 1)]  # Convert to 0-based index
            new_indices = list(range(len(jobs), len(jobs) + len(new_jobs)))
            combined_indices = remaining + new_indices
            reduced_jobs = [all_jobs[i] for i in combined_indices]
            new_order = schedule_jobs(reduced_jobs, m)
            final_sequence = [combined_indices[i] for i in new_order]

            new_completion, new_makespan = calculate_completion_times(all_jobs, final_sequence, m)

            st.subheader("Updated Job Sequence")
            st.write([f"J{i+1}" for i in final_sequence])
            st.write("Makespan:", new_makespan)

            st.subheader("Updated Completion Times")
            st.table(new_completion)

if __name__ == "__main__":
    main()
