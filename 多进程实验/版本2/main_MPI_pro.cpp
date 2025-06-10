#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h" // 假设有md5.h用于哈希计算
#include <iomanip>
#include <unordered_set>
#include <mpi.h>
#include <vector>    // Add for std::vector
#include <iostream>  // For cout/cerr added for clarity
#include <numeric>   // Required for std::accumulate (though not strictly used in final code, good for calculation clarity)
#include <algorithm> // For std::min, std::max (used for conceptual clarity of remainder distribution)

using namespace std;
using namespace chrono;

// 定义MD5Hash函数和bit32类型，如果md5.h没有提供
// 编译命令: mpic++ -o main guessing_MPI_pro.cpp train.cpp main_MPI_pro.cpp md5.cpp

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    double time_hash = 0;             // 每个进程本地的哈希检查时间
    double local_time_guess = 0;      // 每个进程本地的猜测阶段总时间
    double local_pure_guess_time = 0; // 每个进程本地的纯猜测生成时间
    double time_train = 0;
    PriorityQueue q;

    // 训练模型 (只在主进程上执行)
    if (rank == 0)
    {
        ifstream train_file("/guessdata/Rockyou-singleLined-full.txt");
        if (!train_file.is_open())
        {
            cout << "[Rank 0] Error: Cannot open training file /guessdata/Rockyou-singleLined-full.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        train_file.close();

        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order(); // 训练后按概率对PT进行排序，以便分发
        auto end_train = system_clock::now();

        auto duration_train = duration_cast<microseconds>(end_train - start_train);
        time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;
        cout << "[Rank " << rank << "] Training completed, time: " << time_train << " seconds" << endl;
    }

    // 广播模型数据
    cout << "[Rank " << rank << "] Starting broadcast" << endl;
    q.m.broadcast(0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程都接收到模型
    cout << "[Rank " << rank << "] Broadcast completed" << endl;

    // ====================================================================
    // 测试集加载和优先队列初始化、广播
    // ====================================================================

    // 每个进程独立加载测试集，因为其内容相同且是只读访问
    unordered_set<std::string> test_set;
    test_set.reserve(1000000); // 预分配100万元素的空间

    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
    if (!test_data.is_open())
    {
        cout << "[Rank " << rank << "] Error: Cannot open test data file /guessdata/Rockyou-singleLined-full.txt" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int test_count = 0;
    string pw;
    while (test_data >> pw)
    {
        test_count += 1;
        test_set.insert(pw);
        if (test_count >= 1000000) // 只加载前100万个密码
            break;
    }
    test_data.close();

    test_set.rehash(static_cast<size_t>(test_set.size() / test_set.max_load_factor() + 1));

    cout << "[Rank " << rank << "] Test set loaded, size: " << test_set.size()
         << ", load factor: " << test_set.load_factor()
         << ", bucket count: " << test_set.bucket_count() << endl;

    // 只有主进程初始化优先队列并广播
    if (rank == 0)
    {
        q.init(); // 主进程初始化 priority 向量
        cout << "[Rank " << rank << "] Priority queue initialized, total PTs: " << q.priority.size() << endl;
        // 广播 q.priority 给所有其他进程
        q.broadcast_priority_queue(0, MPI_COMM_WORLD);
        cout << "[Rank " << rank << "] Priority queue broadcast completed." << endl;
    }
    else
    {
        // 从属进程接收广播的 q.priority
        q.broadcast_priority_queue(0, MPI_COMM_WORLD);
        cout << "[Rank " << rank << "] Priority queue received, total PTs: " << q.priority.size() << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程都接收到优先队列

    long long local_cracked = 0;
    long long local_total_guesses = 0; // 该进程生成并处理的总猜测数

    const long long GLOBAL_GUESS_LIMIT = 9500000; // 设置全局猜测限制

    auto start_guess_phase = system_clock::now(); // 测量猜测阶段的总时间

    int global_limit_reached_flag = 0; // 用于全局终止的标志

    size_t total_pts = q.priority.size();
    std::vector<size_t> workloads(num_processes);

    // 计算每个进程的基本工作量和剩余PTs
    size_t base_work_per_process = total_pts / num_processes;
    size_t remaining_pts = total_pts % num_processes;

    // 分配基本工作量
    for (int i = 0; i < num_processes; ++i)
    {
        workloads[i] = base_work_per_process;
    }

    // 剩余的PTs以轮询方式均匀分配给前 'remaining_pts' 个进程
    for (int i = 0; i < remaining_pts; ++i)
    {
        workloads[i]++;
    }

    size_t start_idx = 0;
    // 根据前面进程累积的工作量计算当前进程的起始索引
    for (int i = 0; i < rank; ++i)
    {
        start_idx += workloads[i];
    }
    // 结束索引是起始索引加上分配给当前进程的工作量
    size_t end_idx = start_idx + workloads[rank];

    cout << "[Rank " << rank << "] Processing PTs from global index " << start_idx << " to " << end_idx << " (exclusive)."
         << " My calculated workload is " << workloads[rank] << " PTs." << endl;

    // 循环遍历分配给当前进程的PT块
    // 从 start_idx 到 (但不包括) end_idx。
    // current_pt_global_idx 是正在处理的PT的全局索引。
    for (size_t current_pt_global_idx = start_idx; current_pt_global_idx < end_idx; ++current_pt_global_idx)
    {
        // 定期执行全局猜测总数的归约并同步全局限制标志。
        // 这个检查平衡了通信开销和对全局限制的响应速度。
        const int ALLREDUCE_FREQ = 100; // 每 ALLREDUCE_FREQ 个PT执行一次Allreduce
        if ((current_pt_global_idx - start_idx) % ALLREDUCE_FREQ == 0 || current_pt_global_idx == end_idx - 1)
        {
            long long current_global_guesses;
            MPI_Allreduce(&local_total_guesses, &current_global_guesses, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

            if (current_global_guesses >= GLOBAL_GUESS_LIMIT)
            {
                global_limit_reached_flag = 1; // 设置本地标志
            }

            // 确保所有进程在每次迭代中同步全局限制标志，以便快速终止。
            long long temp_limit_reached_ll = global_limit_reached_flag;
            long long global_limit_reached_ll;
            MPI_Allreduce(&temp_limit_reached_ll, &global_limit_reached_ll, 1, MPI_LONG_LONG, MPI_LOR, MPI_COMM_WORLD);
            global_limit_reached_flag = static_cast<int>(global_limit_reached_ll); // 更新全局标志
        }

        // 所有进程检查全局限制标志，以决定是否跳出循环。
        if (global_limit_reached_flag == 1)
        {
            cout << "[Rank " << rank << "] Global guess limit reached. Exiting PT loop early." << endl;
            break; // 同步跳出主循环
        }

        PT current_pt = q.priority[current_pt_global_idx];
        q.guesses.clear(); // 在为当前PT生成新猜测之前清空 guesses

        // 为当前PT生成猜测。
        // Generate函数是每个进程本地的，不涉及MPI通信。
        q.Generate(current_pt);

        bit32 md5_state[4];
        auto start_hash_batch = system_clock::now();
        // 处理当前进程为当前PT生成的猜测。
        for (const string &pw_guess : q.guesses)
        {
            // 可选：在内层循环中进行快速本地检查，以便提前退出。
            // 这可以防止单个进程在下一次MPI_Allreduce同步之前生成过多猜测。
            if (global_limit_reached_flag == 1 || local_total_guesses >= GLOBAL_GUESS_LIMIT)
            {
                break;
            }

            MD5Hash(pw_guess, md5_state);

            if (test_set.count(pw_guess))
            {
                local_cracked += 1;
            }
            local_total_guesses += 1; // 累加该进程处理的总猜测数
        }
        auto end_hash_batch = system_clock::now();
        auto duration_hash_batch = duration_cast<microseconds>(end_hash_batch - start_hash_batch);
        time_hash += double(duration_hash_batch.count()) * microseconds::period::num / microseconds::period::den; // 累加该进程的哈希时间

        q.guesses.clear(); // 清空 guesses，准备下一个PT

        if (rank == 0 && (current_pt_global_idx - start_idx) % 100 == 0 && (current_pt_global_idx - start_idx) > 0)
        {
            cout << "[Rank " << rank << "] Processed local PT index " << (current_pt_global_idx - start_idx)
                 << " (global index " << current_pt_global_idx << "). Local guesses: " << local_total_guesses
                 << ", Local cracked: " << local_cracked << endl;
        }
    } // 本地PT迭代循环结束

    auto end_guess_phase = system_clock::now();
    auto duration_guess_phase = duration_cast<microseconds>(end_guess_phase - start_guess_phase);
    local_time_guess = double(duration_guess_phase.count()) * microseconds::period::num / microseconds::period::den;
    local_pure_guess_time = local_time_guess - time_hash;

    long long global_total_guesses_sum;
    long long global_cracked;
    double global_time_hash_sum;
    double max_program_guess_time;

    MPI_Allreduce(&local_total_guesses, &global_total_guesses_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Reduce(&local_cracked, &global_cracked, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&time_hash, &global_time_hash_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time_guess, &max_program_guess_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "========================================" << endl;
        cout << "[Rank 0] Overall Results:" << endl;
        cout << "[Rank 0] Target Global Guess Limit: " << GLOBAL_GUESS_LIMIT << endl;
        cout << "[Rank 0] Global total guesses generated (actual): " << global_total_guesses_sum << endl;
        cout << "[Rank 0] Global cracked passwords: " << global_cracked << endl;
        cout << "[Rank 0] Max total guess phase time (longest running process): " << fixed << setprecision(6) << max_program_guess_time << " seconds" << endl;
        cout << "[Rank 0] Sum of individual process MD5 hash checking time: " << fixed << setprecision(6) << global_time_hash_sum << " seconds" << endl; // 所有进程哈希时间之和
        cout << "[Rank 0] Training time: " << fixed << setprecision(6) << time_train << " seconds" << endl;
        cout << "GuessTime: " << fixed << setprecision(6) << local_pure_guess_time << " seconds" << endl;
        cout << "========================================" << endl
             << std::flush;
    }

    MPI_Finalize();
    return 0;
}