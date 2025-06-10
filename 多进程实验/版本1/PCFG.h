#ifndef PCFG_H
#define PCFG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <set>
#include <mpi.h>
#include <cstring> // For memcpy

using namespace std;

class segment
{
public:
    int type;   // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values; // Train time only
    unordered_map<int, int> freqs;     // Train time only
    string str_val;                    // This seems to be a remnant, maybe remove or clarify its purpose

    segment(int type, int length) : type(type), length(length) {}
    segment() : type(0), length(0) {} // Default constructor for deserialization

    void insert(string value);
    void order();
    void PrintSeg();
    void PrintValues();

    // 序列化方法
    void serialize(vector<char> &buffer) const
    {
        // type, length, total_freq
        buffer.insert(buffer.end(), (char *)&type, (char *)&type + sizeof(int));
        buffer.insert(buffer.end(), (char *)&length, (char *)&length + sizeof(int));
        buffer.insert(buffer.end(), (char *)&total_freq, (char *)&total_freq + sizeof(int));

        // ordered_values
        size_t size_val = ordered_values.size();
        buffer.insert(buffer.end(), (char *)&size_val, (char *)&size_val + sizeof(size_t));
        for (const auto &str : ordered_values)
        {
            size_t len = str.length();
            buffer.insert(buffer.end(), (char *)&len, (char *)&len + sizeof(size_t));
            buffer.insert(buffer.end(), str.begin(), str.end());
        }

        // ordered_freqs
        size_t size_freq = ordered_freqs.size();
        buffer.insert(buffer.end(), (char *)&size_freq, (char *)&size_freq + sizeof(size_t));
        if (size_freq > 0)
        { // Only copy if there's data
            // Use memcpy directly for better performance for primitive types
            size_t current_buffer_size = buffer.size();
            buffer.resize(current_buffer_size + size_freq * sizeof(int));
            memcpy(buffer.data() + current_buffer_size, ordered_freqs.data(), size_freq * sizeof(int));
        }
    }

    // 反序列化方法
    void deserialize(const char *&buffer)
    {
        memcpy(&type, buffer, sizeof(int));
        buffer += sizeof(int);
        memcpy(&length, buffer, sizeof(int));
        buffer += sizeof(int);
        memcpy(&total_freq, buffer, sizeof(int));
        buffer += sizeof(int);

        // ordered_values
        size_t size_val;
        memcpy(&size_val, buffer, sizeof(size_t));
        buffer += sizeof(size_t);
        ordered_values.clear();
        ordered_values.reserve(size_val); // Pre-allocate to avoid reallocations
        for (size_t i = 0; i < size_val; i++)
        {
            size_t len;
            memcpy(&len, buffer, sizeof(size_t));
            buffer += sizeof(size_t);
            string str(buffer, len);
            buffer += len;
            ordered_values.push_back(str);
        }

        // ordered_freqs
        size_t size_freq;
        memcpy(&size_freq, buffer, sizeof(size_t));
        buffer += sizeof(size_t);
        ordered_freqs.resize(size_freq);
        if (size_freq > 0)
        {
            memcpy(ordered_freqs.data(), buffer, size_freq * sizeof(int));
            buffer += size_freq * sizeof(int);
        }
    }
};

class PT
{
public:
    vector<segment> content;
    int pivot = 0;
    vector<int> curr_indices;
    vector<int> max_indices;
    float preterm_prob;
    float prob;

    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();

    // 序列化方法
    void serialize(vector<char> &buffer) const
    {
        // 序列化content
        size_t size_content = content.size();
        buffer.insert(buffer.end(), (char *)&size_content, (char *)&size_content + sizeof(size_t));
        for (const auto &seg : content)
        {
            seg.serialize(buffer);
        }

        // 序列化curr_indices
        size_t size_curr = curr_indices.size();
        buffer.insert(buffer.end(), (char *)&size_curr, (char *)&size_curr + sizeof(size_t));
        if (size_curr > 0)
        {
            size_t current_buffer_size = buffer.size();
            buffer.resize(current_buffer_size + size_curr * sizeof(int));
            memcpy(buffer.data() + current_buffer_size, curr_indices.data(), size_curr * sizeof(int));
        }

        // 序列化max_indices
        size_t size_max = max_indices.size();
        buffer.insert(buffer.end(), (char *)&size_max, (char *)&size_max + sizeof(size_t));
        if (size_max > 0)
        {
            size_t current_buffer_size = buffer.size();
            buffer.resize(current_buffer_size + size_max * sizeof(int));
            memcpy(buffer.data() + current_buffer_size, max_indices.data(), size_max * sizeof(int));
        }

        // 序列化其他成员
        buffer.insert(buffer.end(), (char *)&prob, (char *)&prob + sizeof(float));
        buffer.insert(buffer.end(), (char *)&preterm_prob, (char *)&preterm_prob + sizeof(float));
        buffer.insert(buffer.end(), (char *)&pivot, (char *)&pivot + sizeof(int));
    }

    // 反序列化方法
    void deserialize(const char *&buffer)
    {
        // 反序列化content
        size_t size_content;
        memcpy(&size_content, buffer, sizeof(size_t));
        buffer += sizeof(size_t);
        content.resize(size_content);
        for (auto &seg : content)
        {
            seg.deserialize(buffer);
        }

        // 反序列化curr_indices
        size_t size_curr;
        memcpy(&size_curr, buffer, sizeof(size_t));
        buffer += sizeof(size_t);
        curr_indices.resize(size_curr);
        if (size_curr > 0)
        {
            memcpy(curr_indices.data(), buffer, size_curr * sizeof(int));
            buffer += size_curr * sizeof(int);
        }

        // 反序列化max_indices
        size_t size_max;
        memcpy(&size_max, buffer, sizeof(size_t));
        buffer += sizeof(size_t);
        max_indices.resize(size_max);
        if (size_max > 0)
        {
            memcpy(max_indices.data(), buffer, size_max * sizeof(int));
            buffer += size_max * sizeof(int);
        }

        // 反序列化其他成员
        memcpy(&prob, buffer, sizeof(float));
        buffer += sizeof(float);
        memcpy(&preterm_prob, buffer, sizeof(float));
        buffer += sizeof(float);
        memcpy(&pivot, buffer, sizeof(int));
        buffer += sizeof(int);
    }
};

class model
{
public:
    int preterm_id = -1, letters_id = -1, digits_id = -1, symbols_id = -1;
    int GetNextPretermID() { return ++preterm_id; }
    int GetNextLettersID() { return ++letters_id; }
    int GetNextDigitsID() { return ++digits_id; }
    int GetNextSymbolsID() { return ++symbols_id; }

    int total_preterm = 0;
    vector<PT> preterminals; // Not directly serialized/deserialized for broadcast in this version
    vector<segment> letters, digits, symbols;
    unordered_map<int, int> preterm_freq;                            // Key is PT index (int), value is frequency
    unordered_map<int, int> letters_freq, digits_freq, symbols_freq; // Not directly serialized/deserialized
    vector<PT> ordered_pts;                                          // This is the list of PTs ordered by probability

    int FindPT(PT pt);
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();

    // 序列化整个模型
    vector<char> serialize() const
    {
        vector<char> buffer;

        // 序列化 letters
        size_t size = letters.size();
        buffer.insert(buffer.end(), (char *)&size, (char *)&size + sizeof(size_t));
        for (const auto &seg : letters)
        {
            seg.serialize(buffer);
        }

        // 序列化 digits
        size = digits.size();
        buffer.insert(buffer.end(), (char *)&size, (char *)&size + sizeof(size_t));
        for (const auto &seg : digits)
        {
            seg.serialize(buffer);
        }

        // 序列化 symbols
        size = symbols.size();
        buffer.insert(buffer.end(), (char *)&size, (char *)&size + sizeof(size_t));
        for (const auto &seg : symbols)
        {
            seg.serialize(buffer);
        }

        // 序列化 total_preterm
        buffer.insert(buffer.end(), (char *)&total_preterm, (char *)&total_preterm + sizeof(int));

        // 序列化 preterm_freq (仅存储 key-value 对)
        size = preterm_freq.size();
        buffer.insert(buffer.end(), (char *)&size, (char *)&size + sizeof(size_t));
        for (const auto &pair : preterm_freq)
        {
            buffer.insert(buffer.end(), (char *)&pair.first, (char *)&pair.first + sizeof(int));
            buffer.insert(buffer.end(), (char *)&pair.second, (char *)&pair.second + sizeof(int));
        }

        return buffer;
    }

    // 反序列化整个模型
    void deserialize(const vector<char> &buffer)
    {
        const char *ptr = buffer.data();

        // 反序列化 letters
        size_t size;
        memcpy(&size, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        letters.resize(size);
        for (auto &seg : letters)
        {
            seg.deserialize(ptr);
        }

        // 反序列化 digits
        memcpy(&size, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        digits.resize(size);
        for (auto &seg : digits)
        {
            seg.deserialize(ptr);
        }

        // 反序列化 symbols
        memcpy(&size, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        symbols.resize(size);
        for (auto &seg : symbols)
        {
            seg.deserialize(ptr);
        }

        // 反序列化 total_preterm
        memcpy(&total_preterm, ptr, sizeof(int));
        ptr += sizeof(int);

        // 反序列化 preterm_freq
        memcpy(&size, ptr, sizeof(size_t));
        ptr += sizeof(size_t);
        preterm_freq.clear();
        for (size_t i = 0; i < size; ++i)
        {
            int key, value;
            memcpy(&key, ptr, sizeof(int));
            ptr += sizeof(int);
            memcpy(&value, ptr, sizeof(int));
            ptr += sizeof(int);
            preterm_freq[key] = value;
        }
    }

    // 通过MPI广播模型
    void broadcast(int root, MPI_Comm comm)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);

        if (rank == root)
        {
            // 主进程序列化并发送模型核心数据
            vector<char> buffer = serialize();
            size_t size = buffer.size();
            MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm);
            MPI_Bcast(buffer.data(), size, MPI_CHAR, root, comm);

            // ordered_pts 在 PriorityQueue::broadcast_priority_queue 中处理，这里不再广播
            // size = ordered_pts.size();
            // MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm);
            // for (const auto &pt : ordered_pts)
            // {
            //     vector<char> pt_buffer;
            //     pt.serialize(pt_buffer);
            //     size_t pt_size = pt_buffer.size();
            //     MPI_Bcast(&pt_size, 1, MPI_UNSIGNED_LONG, root, comm);
            //     MPI_Bcast(pt_buffer.data(), pt_size, MPI_CHAR, root, comm);
            // }
        }
        else
        {
            // 其他进程接收并反序列化模型核心数据
            size_t size;
            MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm);
            vector<char> buffer(size);
            MPI_Bcast(buffer.data(), size, MPI_CHAR, root, comm);
            deserialize(buffer);

            // ordered_pts 在 PriorityQueue::broadcast_priority_queue 中处理，这里不再接收
            // MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm);
            // ordered_pts.resize(size);
            // for (auto &pt : ordered_pts)
            // {
            //     size_t pt_size;
            //     MPI_Bcast(&pt_size, 1, MPI_UNSIGNED_LONG, root, comm);
            //     vector<char> pt_buffer(pt_size);
            //     MPI_Bcast(pt_buffer.data(), pt_size, MPI_CHAR, root, comm);
            //     const char *ptr = pt_buffer.data();
            //     pt.deserialize(ptr);
            // }
        }
    }
};

class PriorityQueue
{
public:
    // 注意：在这个新的并行模型中，priority 向量实际上只是存储了所有 PT 的初始状态
    // 每个进程会根据 rank 独立地从 q.m.ordered_pts 中选择 PT 进行处理
    vector<PT> priority;
    model m;
    vector<string> guesses;      // 用于存储Generate函数生成的猜测
    long long total_guesses = 0; // 单个进程生成的猜测数量

    void CalProb(PT &pt);
    void init();
    void Generate(PT pt); // 修改后的Generate函数，不再包含MPI通信
    // void PopNext(); // 不再需要PopNext，因为每个进程直接迭代处理PT

    // 新增：广播 priority 队列
    void broadcast_priority_queue(int root, MPI_Comm comm)
    {
        int rank;
        MPI_Comm_rank(comm, &rank);

        if (rank == root)
        {
            size_t size = priority.size();
            MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm); // 广播 vector 大小

            for (const auto &pt : priority)
            {
                vector<char> pt_buffer;
                pt.serialize(pt_buffer); // 序列化单个 PT
                size_t pt_size = pt_buffer.size();
                MPI_Bcast(&pt_size, 1, MPI_UNSIGNED_LONG, root, comm);      // 广播单个 PT 的缓冲区大小
                MPI_Bcast(pt_buffer.data(), pt_size, MPI_CHAR, root, comm); // 广播单个 PT 的数据
            }
        }
        else
        {
            size_t size;
            MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm); // 接收 vector 大小
            priority.resize(size);                              // 调整 priority vector 的大小

            for (auto &pt : priority)
            {
                size_t pt_size;
                MPI_Bcast(&pt_size, 1, MPI_UNSIGNED_LONG, root, comm); // 接收单个 PT 的缓冲区大小
                vector<char> pt_buffer(pt_size);
                MPI_Bcast(pt_buffer.data(), pt_size, MPI_CHAR, root, comm); // 接收单个 PT 的数据
                const char *ptr = pt_buffer.data();
                pt.deserialize(ptr); // 反序列化单个 PT
            }
        }
    }
};

#endif