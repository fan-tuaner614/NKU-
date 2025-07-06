#include "PCFG.h"
#include "md5.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mpi.h>
#include <unordered_set>
#include <vector>
#include <string> // 确保包含string头文件

// 假设 MD5Hash_SIMD 已经正确定义
// 例如: void MD5Hash_SIMD(const std::string pw_batch[], uint32_t states_batch[][4], int batch_size);

using namespace std;
using namespace chrono;

void BroadcastSegmentsOptimized(vector<segment> &segments, int root_rank)
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    vector<int> int_buffer;
    vector<char> char_buffer;

    if (world_rank == root_rank)
    {
        int_buffer.push_back(segments.size());
        for (const auto &seg : segments)
        {
            int_buffer.push_back(seg.type);
            int_buffer.push_back(seg.length);
            int_buffer.push_back(seg.total_freq);
            int_buffer.push_back(seg.ordered_values.size());
            int_buffer.push_back(seg.ordered_freqs.size());
            for (const auto &val : seg.ordered_values)
            {
                int_buffer.push_back(val.length());
                char_buffer.insert(char_buffer.end(), val.begin(), val.end());
            }
            int_buffer.insert(int_buffer.end(), seg.ordered_freqs.begin(), seg.ordered_freqs.end());
        }
    }

    long long int_buffer_size = (world_rank == root_rank) ? int_buffer.size() : 0;
    MPI_Bcast(&int_buffer_size, 1, MPI_LONG_LONG, root_rank, MPI_COMM_WORLD);
    if (world_rank != root_rank)
        int_buffer.resize(int_buffer_size);
    MPI_Bcast(int_buffer.data(), int_buffer_size, MPI_INT, root_rank, MPI_COMM_WORLD);

    long long char_buffer_size = (world_rank == root_rank) ? char_buffer.size() : 0;
    MPI_Bcast(&char_buffer_size, 1, MPI_LONG_LONG, root_rank, MPI_COMM_WORLD);
    if (world_rank != root_rank)
        char_buffer.resize(char_buffer_size);
    MPI_Bcast(char_buffer.data(), char_buffer_size, MPI_CHAR, root_rank, MPI_COMM_WORLD);

    if (world_rank != root_rank)
    {
        segments.clear();
        if (int_buffer.empty())
            return;
        size_t int_idx = 0;
        size_t char_idx = 0;
        int num_segments = int_buffer[int_idx++];
        segments.resize(num_segments);
        for (int i = 0; i < num_segments; ++i)
        {
            segments[i].type = int_buffer[int_idx++];
            segments[i].length = int_buffer[int_idx++];
            segments[i].total_freq = int_buffer[int_idx++];
            int val_count = int_buffer[int_idx++];
            int freq_count = int_buffer[int_idx++];
            segments[i].ordered_values.resize(val_count);
            segments[i].ordered_freqs.resize(freq_count);
            for (int j = 0; j < val_count; ++j)
            {
                int len = int_buffer[int_idx++];
                if (len > 0)
                {
                    if (char_idx + len > char_buffer.size())
                    {
                        //cerr << "Rank " << world_rank << ": Fatal error during deserialization. Char buffer overflow." << endl;
                        MPI_Abort(MPI_COMM_WORLD, -1);
                    }
                    segments[i].ordered_values[j].assign(&char_buffer[char_idx], len);
                    char_idx += len;
                }
            }
            for (int j = 0; j < freq_count; ++j)
            {
                segments[i].ordered_freqs[j] = int_buffer[int_idx++];
            }
        }
    }
}

void BroadcastModel(PriorityQueue &q, int root_rank)
{
    BroadcastSegmentsOptimized(q.m.letters, root_rank);
    BroadcastSegmentsOptimized(q.m.digits, root_rank);
    BroadcastSegmentsOptimized(q.m.symbols, root_rank);
}

void BroadcastPT(PT &pt, int root_rank)
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int num_seg = (world_rank == root_rank) ? pt.content.size() : 0;
    MPI_Bcast(&num_seg, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    if (world_rank != root_rank)
        pt.content.resize(num_seg);
    for (int i = 0; i < num_seg; ++i)
    {
        int type = 0, length = 0;
        if (world_rank == root_rank)
        {
            type = pt.content[i].type;
            length = pt.content[i].length;
        }
        MPI_Bcast(&type, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        MPI_Bcast(&length, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        if (world_rank != root_rank)
        {
            pt.content[i].type = type;
            pt.content[i].length = length;
        }
    }
    auto broadcast_vec = [&](vector<int> &vec)
    {
        int size = (world_rank == root_rank) ? vec.size() : 0;
        MPI_Bcast(&size, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        if (world_rank != root_rank)
            vec.resize(size);
        if (size > 0)
            MPI_Bcast(vec.data(), size, MPI_INT, root_rank, MPI_COMM_WORLD);
    };
    broadcast_vec(pt.curr_indices);
    broadcast_vec(pt.max_indices);
}

void BroadcastTestSet(unordered_set<string> &test_set, int root_rank)
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    vector<string> temp_vec;
    if (world_rank == root_rank)
        temp_vec = vector<string>(test_set.begin(), test_set.end());
    int size = temp_vec.size();
    MPI_Bcast(&size, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i)
    {
        int len = (world_rank == root_rank) ? temp_vec[i].size() : 0;
        MPI_Bcast(&len, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
        vector<char> buffer(len);
        if (world_rank == root_rank)
            copy(temp_vec[i].begin(), temp_vec[i].end(), buffer.begin());
        MPI_Bcast(buffer.data(), len, MPI_CHAR, root_rank, MPI_COMM_WORLD);
        if (world_rank != root_rank)
            test_set.insert(string(buffer.begin(), buffer.end()));
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    PriorityQueue q;
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    if (world_rank == 0)
    {
        auto start_train = system_clock::now();
        q.m.train("/guessdata/Rockyou-singleLined-full.txt");
        q.m.order();
        auto end_train = system_clock::now();
        time_train = duration_cast<microseconds>(end_train - start_train).count() * 1e-6;
        //cout << "Training completed in " << time_train << " seconds" << endl
         //    << flush;
    }

    BroadcastModel(q, 0);

    unordered_set<string> test_set;
    if (world_rank == 0)
    {
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        int test_count = 0;
        string pw;
        while (test_data >> pw && test_count < 1000000)
        {
            test_set.insert(pw);
            test_count++;
        }
        //cout << "Rank 0: Test set size = " << test_set.size() << endl
        //     << flush;
    }
    BroadcastTestSet(test_set, 0);

    if (world_rank == 0)
        q.init();

    int total_cracked = 0; // 累加所有循环的破解计数
    int history = 0;
    auto start = system_clock::now();

    int should_exit = 0;
    while (!should_exit)
    {
        int keep_going = 1;
        PT top_pt;
        if (world_rank == 0)
        {
            if (q.priority.empty())
            {
                keep_going = 0;
            }
            else
            {
                top_pt = q.priority.top();
                q.priority.pop();
            }
        }
        MPI_Bcast(&keep_going, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!keep_going)
        {
            should_exit = 1;
            break;
        }

        BroadcastPT(top_pt, 0);

        vector<string> local_guesses;
        auto start_guess = system_clock::now();
        q.Generate(top_pt, local_guesses);
        auto end_guess = system_clock::now();
        time_guess += duration_cast<microseconds>(end_guess - start_guess).count() * 1e-6;

        //cout << "Rank " << world_rank << ": local_guesses.size() = " << local_guesses.size() << endl
         //    << flush;

        // SIMD 优化后的哈希处理
        auto start_hash_loop = system_clock::now(); // 使用新的变量名避免混淆
        int local_cracked = 0;
        const int BATCH_SIZE = 4;
        string pw_batch[BATCH_SIZE]; // 批处理的口令字符串数组
        // 使用 local_guesses.size() 代替 q.guesses.size()
        const int aligned_size = (local_guesses.size() / BATCH_SIZE) * BATCH_SIZE;
        // 使用 uint32_t 代替 bit32，并确保对齐
        alignas(16) uint32_t states[BATCH_SIZE][4]; 

        for (int i = 0; i < aligned_size; i += BATCH_SIZE)
        {
            for (int j = 0; j < BATCH_SIZE; ++j)
            {
                pw_batch[j] = local_guesses[i + j];
                // 在批处理循环中进行破解检查
                if (test_set.find(pw_batch[j]) != test_set.end())
                {
                    local_cracked++;
                }
            }
            MD5Hash_SIMD(pw_batch, states, BATCH_SIZE);
        }

        // 处理剩余元素
        for (int i = aligned_size; i < local_guesses.size(); ++i)
        {
            // 对于剩余元素进行破解检查
            if (test_set.find(local_guesses[i]) != test_set.end())
            {
                local_cracked++;
            }
            bit32 state[4]; // 单个哈希的状态变量
            MD5Hash(local_guesses[i], state); // 原始单次处理
        }
        auto end_hash_loop = system_clock::now();
        time_hash += duration_cast<microseconds>(end_hash_loop - start_hash_loop).count() * 1e-6;

        //cout << "Rank " << world_rank << ": local_cracked = " << local_cracked << endl
         //    << flush;

        int local_count = local_guesses.size();
        int global_count = 0;
        int cracked = 0; // 单次循环的破解计数
        MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_cracked, &cracked, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (world_rank == 0)
        {
            history += global_count;
            total_cracked += cracked; // 累加每次循环的破解计数
            cout << "Guesses generated: " << history << endl
                << flush;
            if (history >= 10000000)
            {
                should_exit = 1;
            }
        }
        MPI_Bcast(&should_exit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    auto end = system_clock::now();
    time_guess = duration_cast<microseconds>(end - start).count() * 1e-6;

    if (world_rank == 0)
    {
        cout << "Guess time: " << time_guess - time_hash << " seconds" << endl
             << flush;
        cout << "Hash time: " << time_hash << " seconds" << endl
             << flush;
        cout << "Train time: " << time_train << " seconds" << endl
             << flush;
        cout << "Cracked: " << total_cracked << endl
             << flush;
    }

    MPI_Barrier(MPI_COMM_WORLD); // 确保所有进程同步
    MPI_Finalize();
    return 0;
}