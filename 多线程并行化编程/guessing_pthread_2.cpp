#include "PCFG_pthread_2.h"
#include <pthread.h>
#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <algorithm>
#include <atomic>
#include <unistd.h>

using namespace std;

// ThreadData 结构（基于之前提供的内容）
struct ThreadData
{
    int start;
    int end;
    segment *a;
    vector<string> *local_guesses;
    string base_guess;
    bool is_single_segment;
    int thread_id;  // 线程编号
    int task_count; // 任务计数
};

// ThreadPool 结构
struct ThreadPool
{
    pthread_t *threads;
    queue<ThreadData> *tasks;
    pthread_mutex_t task_mutex;
    pthread_cond_t task_cond;
    pthread_cond_t completion_cond;
    bool stop;
    int num_threads;
    atomic<int> active_tasks;
};

// 初始化线程池
ThreadPool *thread_pool_init(size_t num_threads)
{
    ThreadPool *pool = new ThreadPool;
    pool->num_threads = num_threads;
    pool->tasks = new queue<ThreadData>();
    pool->stop = false;
    pool->active_tasks = 0;

    pthread_mutex_init(&pool->task_mutex, nullptr);
    pthread_cond_init(&pool->task_cond, nullptr);
    pthread_cond_init(&pool->completion_cond, nullptr);

    pool->threads = new pthread_t[num_threads];
    for (size_t i = 0; i < num_threads; ++i)
    {
        pthread_create(&pool->threads[i], nullptr, [](void *arg) -> void *
                       {
            ThreadPool* pool = static_cast<ThreadPool*>(arg);
            while (true) {
                ThreadData task;
                pthread_mutex_lock(&pool->task_mutex);
                while (pool->tasks->empty() && !pool->stop) {
                    pthread_cond_wait(&pool->task_cond, &pool->task_mutex);
                }
                if (pool->stop && pool->tasks->empty()) {
                    pthread_mutex_unlock(&pool->task_mutex);
                    return nullptr;
                }
                task = pool->tasks->front();
                pool->tasks->pop();
                pool->active_tasks++;
                pthread_mutex_unlock(&pool->task_mutex);

                // 执行任务
                if (task.is_single_segment) {
                    for (int i = task.start; i < task.end; ++i) {
                        task.local_guesses->push_back(task.a->ordered_values[i]);
                    }
                } else {
                    for (int i = task.start; i < task.end; ++i) {
                        string guess = task.base_guess + task.a->ordered_values[i];
                        task.local_guesses->push_back(guess);
                    }
                }

                // 信号任务完成
                pthread_mutex_lock(&pool->task_mutex);
                pool->active_tasks--;
                pthread_cond_signal(&pool->completion_cond);
                pthread_mutex_unlock(&pool->task_mutex);
            }
            return nullptr; }, pool);
    }
    return pool;
}

// 销毁线程池
void thread_pool_destroy(ThreadPool *pool)
{
    pthread_mutex_lock(&pool->task_mutex);
    pool->stop = true;
    pthread_cond_broadcast(&pool->task_cond);
    pthread_mutex_unlock(&pool->task_mutex);

    for (size_t i = 0; i < pool->num_threads; ++i)
    {
        pthread_join(pool->threads[i], nullptr);
    }

    delete[] pool->threads;
    delete pool->tasks;
    pthread_mutex_destroy(&pool->task_mutex);
    pthread_cond_destroy(&pool->task_cond);
    pthread_cond_destroy(&pool->completion_cond);
    delete pool;
}

// 入队任务
void thread_pool_enqueue(ThreadPool *pool, const ThreadData &task)
{
    pthread_mutex_lock(&pool->task_mutex);
    pool->tasks->push(task);
    pthread_cond_signal(&pool->task_cond);
    pthread_mutex_unlock(&pool->task_mutex);
}

// 等待所有任务完成
void thread_pool_wait_all(ThreadPool *pool)
{
    pthread_mutex_lock(&pool->task_mutex);
    while (pool->active_tasks > 0 || !pool->tasks->empty())
    {
        pthread_cond_wait(&pool->completion_cond, &pool->task_mutex);
    }
    pthread_mutex_unlock(&pool->task_mutex);
}

// PriorityQueue 实现
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    vector<vector<string>> local_guesses_per_thread(pool->num_threads); // 每个线程的局部猜测存储

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[0])];
        }
        else if (pt.content[0].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[0])];
        }
        else if (pt.content[0].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }

        int total_indices = pt.max_indices[0];
        int indices_per_thread = (total_indices + pool->num_threads - 1) / pool->num_threads;

        for (int t = 0; t < pool->num_threads; ++t)
        {
            ThreadData task;
            task.start = t * indices_per_thread;
            task.end = min((t + 1) * indices_per_thread, total_indices);
            task.a = a;
            task.local_guesses = &local_guesses_per_thread[t];
            task.base_guess = "";
            task.is_single_segment = true;
            task.thread_id = t;
            task.task_count = task.end - task.start;

            if (task.start < task.end)
            {
                thread_pool_enqueue(pool, task);
            }
        }
    }
    else
    {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
            {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 2)
            {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            }
            else if (pt.content[seg_idx].type == 3)
            {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx++;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a;
        if (pt.content[pt.content.size() - 1].type == 1)
        {
            a = &m.letters[m.FindLetter(pt.content[pt.content.size() - 1])];
        }
        else if (pt.content[pt.content.size() - 1].type == 2)
        {
            a = &m.digits[m.FindDigit(pt.content[pt.content.size() - 1])];
        }
        else if (pt.content[pt.content.size() - 1].type == 3)
        {
            a = &m.symbols[m.FindSymbol(pt.content[pt.content.size() - 1])];
        }

        int total_indices = pt.max_indices[pt.content.size() - 1];
        int indices_per_thread = (total_indices + pool->num_threads - 1) / pool->num_threads;

        for (int t = 0; t < pool->num_threads; ++t)
        {
            ThreadData task;
            task.start = t * indices_per_thread;
            task.end = min((t + 1) * indices_per_thread, total_indices);
            task.a = a;
            task.local_guesses = &local_guesses_per_thread[t];
            task.base_guess = guess;
            task.is_single_segment = false;
            task.thread_id = t;
            task.task_count = task.end - task.start;

            if (task.start < task.end)
            {
                thread_pool_enqueue(pool, task);
            }
        }
    }

    // 等待所有任务完成
    thread_pool_wait_all(pool);

    // 合并所有线程的局部猜测结果到全局 guesses
    for (int t = 0; t < pool->num_threads; ++t)
    {
        guesses.insert(guesses.end(),
                       local_guesses_per_thread[t].begin(),
                       local_guesses_per_thread[t].end());
        total_guesses += local_guesses_per_thread[t].size();
    }
}

PriorityQueue::PriorityQueue()
{
    total_guesses = 0;
    int num_threads = 8;
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    if (nprocs > 0)
    {
        num_threads = nprocs;
    }
    pool = thread_pool_init(num_threads);
}

PriorityQueue::~PriorityQueue()
{
    thread_pool_destroy(pool);
}

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
            {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            else if (seg.type == 2)
            {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            else if (seg.type == 3)
            {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
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
    {
        return res;
    }
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
