#include "PCFG.h"
#include <cstring>
#include <vector>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>

// 优化后的kernel：用offsets和all_data定位每个字符串
__global__ void generate_guesses_kernel(const char *all_data, const int *offsets, int num, char *d_output, int max_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num) {
        int start = offsets[idx];
        int len = offsets[idx + 1] - start;
        const char *src = all_data + start;
        char *dst = d_output + idx * max_len;

        // 使用memcpy替代手动循环
        if (len > 0) {
            memcpy(dst, src, len);
        }
        dst[len] = '\0';
    }
}

// 可选：向量化拷贝（仅当max_len和src对齐时才有意义）
// __global__ void generate_guesses_kernel_vec(const char *all_data, const int *offsets, int num, char *d_output, int max_len) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < num) {
//         int start = offsets[idx];
//         int len = offsets[idx + 1] - start;
//         const char *src = all_data + start;
//         char *dst = d_output + idx * max_len;
//         int i = 0;
//         for (; i + 3 < len; i += 4) {
//             *(reinterpret_cast<int*>(dst + i)) = *(reinterpret_cast<const int*>(src + i));
//         }
//         for (; i < len; ++i) dst[i] = src[i];
//         dst[len] = '\0';
//     }
// }
#endif

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt); // 修正
    }
}

void PriorityQueue::PopNext()
{
    // 适配std::vector
    if (priority.empty()) return;
    Generate(priority.front());
    PT pt_top = priority.front(); // 先拷贝一份
    vector<PT> new_pts = pt_top.NewPTs();
    priority.erase(priority.begin());
    for (PT pt : new_pts)
    {
        CalProb(pt);
        priority.emplace_back(pt); // 修正
    }
}

void PriorityQueue::PopNextParallel() 
{
    int batch_size = 100;
    if (priority.empty()) return;
    int n = std::min(batch_size, (int)priority.size());
    std::vector<PT> batch_pts(priority.begin(), priority.begin() + n);

    int max_len = 0;
    std::vector<std::string> all_guesses;
    std::vector<int> offsets;
    offsets.push_back(0);

    for (PT& pt : batch_pts) {
        // 计算概率
        CalProb(pt);

        segment *a;
        std::string guess_prefix;
        int num;
        if (pt.content.size() == 1) {
            if (pt.content[0].type == 1)
                a = &m.letters[m.FindLetter(pt.content[0])];
            else if (pt.content[0].type == 2)
                a = &m.digits[m.FindDigit(pt.content[0])];
            else
                a = &m.symbols[m.FindSymbol(pt.content[0])];
            num = pt.max_indices[0];
            for (int i = 0; i < num; ++i) {
                const std::string& s = a->ordered_values[i];
                all_guesses.push_back(s);
                offsets.push_back(offsets.back() + s.size());
                if (s.size() > max_len) max_len = s.size();
            }
        } else {
            int seg_idx = 0;
            for (int idx : pt.curr_indices) {
                if (pt.content[seg_idx].type == 1)
                    guess_prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
                else if (pt.content[seg_idx].type == 2)
                    guess_prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
                else
                    guess_prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
                seg_idx += 1;
                if (seg_idx == pt.content.size() - 1)
                    break;
            }
            if (pt.content[pt.content.size() - 1].type == 1)
                a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
            else if (pt.content[pt.content.size() - 1].type == 2)
                a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
            else
                a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
            num = pt.max_indices[pt.content.size() - 1];
            for (int i = 0; i < num; ++i) {
                std::string temp = guess_prefix + a->ordered_values[i];
                all_guesses.push_back(temp);
                offsets.push_back(offsets.back() + temp.size());
                if (temp.size() > max_len) max_len = temp.size();
            }
        }
    }
    int total_guesses = all_guesses.size();
    max_len += 1;

#ifdef __CUDACC__
    if (total_guesses > 10000) {
        // 拼接所有字符串到一块连续内存
        std::vector<char> all_data(offsets.back());
        int pos = 0;
        for (const auto& s : all_guesses) {
            memcpy(&all_data[pos], s.data(), s.size());
            pos += s.size();
        }

        static char *d_all_data = nullptr;
        static int *d_offsets = nullptr;
        static char *d_output = nullptr;
        static size_t all_data_capacity = 0, offsets_capacity = 0, output_capacity = 0;
        static cudaStream_t stream = nullptr;
        if (!stream) cudaStreamCreate(&stream);

        if (all_data.size() > all_data_capacity) {
            if (d_all_data) cudaFree(d_all_data);
            cudaMalloc(&d_all_data, all_data.size() * sizeof(char));
            all_data_capacity = all_data.size();
        }
        if (offsets.size() > offsets_capacity) {
            if (d_offsets) cudaFree(d_offsets);
            cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
            offsets_capacity = offsets.size();
        }
        if ((total_guesses * max_len) > output_capacity) {
            if (d_output) cudaFree(d_output);
            cudaMalloc(&d_output, total_guesses * max_len * sizeof(char));
            output_capacity = total_guesses * max_len;
        }

        cudaMemcpyAsync(d_all_data, all_data.data(), all_data.size() * sizeof(char), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

        int block = 256;
        int grid = (total_guesses + block - 1) / block;
        generate_guesses_kernel<<<grid, block, 0, stream>>>(d_all_data, d_offsets, total_guesses, d_output, max_len);

        static std::vector<char> h_output;
        h_output.resize(total_guesses * max_len);
        cudaMemcpyAsync(h_output.data(), d_output, total_guesses * max_len * sizeof(char), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        guesses.reserve(guesses.size() + total_guesses);
        for (int i = 0; i < total_guesses; ++i) {
            guesses.emplace_back(&h_output[i * max_len]);
            this->total_guesses += 1;
        }
    } else
#endif
    {
        // CPU串行处理
        for (const auto& s : all_guesses) {
            guesses.emplace_back(s);
            this->total_guesses += 1;
        }
    }

    // 生成新PT并加入队列
    for (PT &pt : batch_pts) {
        vector<PT> new_pts = pt.NewPTs();
        for (PT &new_pt : new_pts) {
            CalProb(new_pt);
            priority.emplace_back(new_pt);
        }
    }
    priority.erase(priority.begin(), priority.begin() + n);
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
    {
        return res;
    }
    else
    {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i += 1)
        {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i])
            {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
    return res;
}


void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int num = pt.max_indices[0];
        int max_len = 0;
        for (int i = 0; i < num; ++i)
            if (a->ordered_values[i].size() > max_len)
                max_len = a->ordered_values[i].size();
        max_len += 1;
        //std::cout<<num<<endl;
        // 判断任务量，选择CPU或GPU
        if (num <= 1000000) {
            // CPU串行实现（参考guessing.cpp）
            for (int i = 0; i < num; i += 1)
            {
                string guess = a->ordered_values[i];
                guesses.emplace_back(guess);
                total_guesses += 1;
            }
        }
#ifdef __CUDACC__
        else
        {
            std::cout<<"GPU"<<endl;
            // 静态内存复用（仅示例，线程安全需注意）
            static char *d_all_data = nullptr;
            static int *d_offsets = nullptr;
            static char *d_output = nullptr;
            static size_t all_data_capacity = 0, offsets_capacity = 0, output_capacity = 0;
            static cudaStream_t stream = nullptr;
            if (!stream) cudaStreamCreate(&stream);

            // 拼接所有字符串到一块连续内存，并记录offsets
            std::vector<char> all_data;
            std::vector<int> offsets(num + 1);
            int pos = 0;
            for (int i = 0; i < num; ++i) {
                offsets[i] = pos;
                const std::string &s = a->ordered_values[i];
                all_data.insert(all_data.end(), s.begin(), s.end());
                pos += s.size();
            }
            offsets[num] = pos;

            // 分配或复用 device 内存
            if (all_data.size() > all_data_capacity) {
                if (d_all_data) cudaFree(d_all_data);
                cudaMalloc(&d_all_data, all_data.size() * sizeof(char));
                all_data_capacity = all_data.size();
            }
            if ((num + 1) > offsets_capacity) {
                if (d_offsets) cudaFree(d_offsets);
                cudaMalloc(&d_offsets, (num + 1) * sizeof(int));
                offsets_capacity = num + 1;
            }
            if ((num * max_len) > output_capacity) {
                if (d_output) cudaFree(d_output);
                cudaMalloc(&d_output, num * max_len * sizeof(char));
                output_capacity = num * max_len;
            }

            // 异步拷贝
            cudaMemcpyAsync(d_all_data, all_data.data(), all_data.size() * sizeof(char), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_offsets, offsets.data(), (num + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

            int block = 256;
            int grid = (num + block - 1) / block;
            generate_guesses_kernel<<<grid, block, 0, stream>>>(d_all_data, d_offsets, num, d_output, max_len);

            // 异步拷贝回主机
            static std::vector<char> h_output;
            h_output.resize(num * max_len);
            cudaMemcpyAsync(h_output.data(), d_output, num * max_len * sizeof(char), cudaMemcpyDeviceToHost, stream);

            // 只在需要时同步
            cudaStreamSynchronize(stream);

            guesses.reserve(guesses.size() + num);
            for (int i = 0; i < num; ++i) {
                guesses.emplace_back(&h_output[i * max_len]);
                total_guesses += 1;
            }
        }
#endif
    }
    else
    {
        string guess_prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                guess_prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 2)
                guess_prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            if (pt.content[seg_idx].type == 3)
                guess_prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 2)
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        if (pt.content[pt.content.size() - 1].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];

        int num = pt.max_indices[pt.content.size() - 1];
        int max_len = guess_prefix.size();
        for (int i = 0; i < num; ++i)
            if (a->ordered_values[i].size() + guess_prefix.size() > max_len)
                max_len = a->ordered_values[i].size() + guess_prefix.size();
        max_len += 1;
        //std::cout<<num<<endl;
        // 判断任务量，选择CPU或GPU
        if (num <= 1000000) {
            // CPU串行实现（参考guessing.cpp）
            for (int i = 0; i < num; i += 1)
            {
                string temp = guess_prefix + a->ordered_values[i];
                guesses.emplace_back(temp);
                total_guesses += 1;
            }
        }
#ifdef __CUDACC__
        else
        {
            std::cout<<"GPU"<<endl;
            // 拼接所有字符串到一块连续内存，并记录offsets
            std::vector<char> all_data;
            std::vector<int> offsets(num + 1);
            int pos = 0;
            for (int i = 0; i < num; ++i) {
                std::string temp = guess_prefix + a->ordered_values[i];
                offsets[i] = pos;
                all_data.insert(all_data.end(), temp.begin(), temp.end());
                pos += temp.size();
            }
            offsets[num] = pos;

            // 静态内存复用（仅示例，线程安全需注意）
            static char *d_all_data = nullptr;
            static int *d_offsets = nullptr;
            static char *d_output = nullptr;
            static size_t all_data_capacity = 0, offsets_capacity = 0, output_capacity = 0;
            static cudaStream_t stream = nullptr;
            if (!stream) cudaStreamCreate(&stream);

            // 分配或复用 device 内存
            if (all_data.size() > all_data_capacity) {
                if (d_all_data) cudaFree(d_all_data);
                cudaMalloc(&d_all_data, all_data.size() * sizeof(char));
                all_data_capacity = all_data.size();
            }
            if ((num + 1) > offsets_capacity) {
                if (d_offsets) cudaFree(d_offsets);
                cudaMalloc(&d_offsets, (num + 1) * sizeof(int));
                offsets_capacity = num + 1;
            }
            if ((num * max_len) > output_capacity) {
                if (d_output) cudaFree(d_output);
                cudaMalloc(&d_output, num * max_len * sizeof(char));
                output_capacity = num * max_len;
            }

            // 异步拷贝
            cudaMemcpyAsync(d_all_data, all_data.data(), all_data.size() * sizeof(char), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_offsets, offsets.data(), (num + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

            int block = 256;
            int grid = (num + block - 1) / block;
            generate_guesses_kernel<<<grid, block, 0, stream>>>(d_all_data, d_offsets, num, d_output, max_len);

            // 异步拷贝回主机
            static std::vector<char> h_output;
            h_output.resize(num * max_len);
            cudaMemcpyAsync(h_output.data(), d_output, num * max_len * sizeof(char), cudaMemcpyDeviceToHost, stream);

            // 只在需要时同步
            cudaStreamSynchronize(stream);

            guesses.reserve(guesses.size() + num);
            for (int i = 0; i < num; ++i) {
                guesses.emplace_back(&h_output[i * max_len]);
                total_guesses += 1;
            }
        }
#endif
    }
}