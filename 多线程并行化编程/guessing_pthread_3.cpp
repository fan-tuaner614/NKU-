#include "PCFG.h"
#include <pthread.h>
#include <vector>
#include <string>
#include <atomic>
#include <algorithm>
#include <queue>
#include <unistd.h>
#include <iostream>
// 0.42s-
using namespace std;

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    int problem_size = (pt.content.size() == 1)
                           ? pt.max_indices[0]
                           : pt.max_indices[pt.content.size() - 1];

    if (problem_size < 10000)
    {
        // 串行处理，直接参考guessing.cpp中的实现
        if (pt.content.size() == 1)
        {
            segment *a = nullptr;
            switch (pt.content[0].type)
            {
            case 1:
                a = &m.letters[m.FindLetter(pt.content[0])];
                break;
            case 2:
                a = &m.digits[m.FindDigit(pt.content[0])];
                break;
            case 3:
                a = &m.symbols[m.FindSymbol(pt.content[0])];
                break;
            }
            for (int i = 0; i < problem_size; ++i)
            {
                guesses.push_back(a->ordered_values[i]);
                total_guesses++;
            }
        }
        else
        {
            string base_guess;
            int seg_idx = 0;
            for (int idx : pt.curr_indices)
            {
                if (seg_idx >= pt.content.size() - 1)
                    break;
                const segment &seg = pt.content[seg_idx];
                switch (seg.type)
                {
                case 1:
                    base_guess += m.letters[m.FindLetter(seg)].ordered_values[idx];
                    break;
                case 2:
                    base_guess += m.digits[m.FindDigit(seg)].ordered_values[idx];
                    break;
                case 3:
                    base_guess += m.symbols[m.FindSymbol(seg)].ordered_values[idx];
                    break;
                }
                seg_idx++;
            }
            segment *a = nullptr;
            const segment &last_seg = pt.content.back();
            switch (last_seg.type)
            {
            case 1:
                a = &m.letters[m.FindLetter(last_seg)];
                break;
            case 2:
                a = &m.digits[m.FindDigit(last_seg)];
                break;
            case 3:
                a = &m.symbols[m.FindSymbol(last_seg)];
                break;
            }
            for (int i = 0; i < problem_size; ++i)
            {
                guesses.push_back(base_guess + a->ordered_values[i]);
                total_guesses++;
            }
        }
    }
    else
    {
        // 固定8线程，主线程也参与计算
        const int num_threads = 8;
        segment *a = nullptr;
        string base_guess;
        bool is_single_segment = (pt.content.size() == 1);

        if (is_single_segment)
        {
            const segment &seg = pt.content[0];
            switch (seg.type)
            {
            case 1:
                a = &m.letters[m.FindLetter(seg)];
                break;
            case 2:
                a = &m.digits[m.FindDigit(seg)];
                break;
            case 3:
                a = &m.symbols[m.FindSymbol(seg)];
                break;
            }
        }
        else
        {
            int seg_idx = 0;
            for (int idx : pt.curr_indices)
            {
                if (seg_idx >= pt.content.size() - 1)
                    break;
                const segment &seg = pt.content[seg_idx];
                switch (seg.type)
                {
                case 1:
                    base_guess += m.letters[m.FindLetter(seg)].ordered_values[idx];
                    break;
                case 2:
                    base_guess += m.digits[m.FindDigit(seg)].ordered_values[idx];
                    break;
                case 3:
                    base_guess += m.symbols[m.FindSymbol(seg)].ordered_values[idx];
                    break;
                }
                seg_idx++;
            }
            const segment &last_seg = pt.content.back();
            switch (last_seg.type)
            {
            case 1:
                a = &m.letters[m.FindLetter(last_seg)];
                break;
            case 2:
                a = &m.digits[m.FindDigit(last_seg)];
                break;
            case 3:
                a = &m.symbols[m.FindSymbol(last_seg)];
                break;
            }
        }

        int total_iterations = problem_size;
        int chunk_size = (total_iterations + num_threads - 1) / num_threads;
        size_t old_size = guesses.size();
        guesses.resize(old_size + total_iterations);

        struct RangeThreadData
        {
            int start, end;
            segment *a;
            string base_guess;
            bool is_single_segment;
            vector<string> *guesses;
            size_t offset;
            int thread_id;  // 设置线程编号
            int task_count; // 设置任务数量
        };

        auto thread_func = [](void *arg) -> void *
        {
            RangeThreadData *data = static_cast<RangeThreadData *>(arg);
            // cout << "Thread " << data->thread_id << " started, processing "
            //     << data->task_count << " tasks" << endl;

            if (data->is_single_segment)
            {
                for (int i = data->start; i < data->end; ++i)
                {
                    (*data->guesses)[data->offset + i - data->start] = data->a->ordered_values[i];
                }
            }
            else
            {
                for (int i = data->start; i < data->end; ++i)
                {
                    (*data->guesses)[data->offset + i - data->start] = data->base_guess + data->a->ordered_values[i];
                }
            }

            // cout << "Thread " << data->thread_id << " completed "
            //     << data->task_count << " tasks" << endl;
            return nullptr;
        };

        pthread_t threads[num_threads - 1];
        RangeThreadData thread_data[num_threads];

        // 预先计算所有线程的任务范围
        for (int t = 0; t < num_threads; ++t)
        {
            thread_data[t].start = t * chunk_size;
            thread_data[t].end = std::min((t + 1) * chunk_size, total_iterations);
            thread_data[t].a = a;
            thread_data[t].base_guess = base_guess;
            thread_data[t].is_single_segment = is_single_segment;
            thread_data[t].guesses = &guesses;
            thread_data[t].offset = old_size + thread_data[t].start;
            thread_data[t].thread_id = t;
            thread_data[t].task_count = thread_data[t].end - thread_data[t].start;
        }

        // 1. 主线程先启动所有工作线程
        for (int t = 1; t < num_threads; ++t)
        {
            if (thread_data[t].task_count > 0)
            {
                pthread_create(&threads[t - 1], nullptr, thread_func, &thread_data[t]);
            }
        }

        // 2. 主线程处理自己的任务
        thread_func(&thread_data[0]);

        // 3. 等待其他线程完成
        for (int t = 1; t < num_threads; ++t)
        {
            if (thread_data[t].task_count > 0)
            {
                pthread_join(threads[t - 1], nullptr);
            }
        }

        total_guesses += total_iterations;
    }
}

// Unchanged functions (for completeness, not modified)
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
        else if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        else if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index++;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            else if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            else if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext()
{
    Generate(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                {
                    priority.emplace(iter + 1, pt);
                    break;
                }
            }
            else if (iter == priority.end() - 1)
            {
                priority.emplace_back(pt);
                break;
            }
            else if (iter == priority.begin() && iter->prob < pt.prob)
            {
                priority.emplace(iter, pt);
                break;
            }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1)
        return res;

    int init_pivot = pivot;
    for (int i = pivot; i < curr_indices.size() - 1; i++)
    {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i])
        {
            pivot = i;
            res.emplace_back(*this);
        }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}