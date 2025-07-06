#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
using namespace std;
// 这个文件里面的各函数你都不需要完全理解，甚至根本不需要看
// 从学术价值上讲，加速模型的训练过程是一个没什么价值的问题，因为我们一般假定统计学模型的训练成本较低
// 但是，假如你是一个投稿时顶着ddl做实验的倒霉研究生/实习生，提高训练速度就可以大幅节省你的时间了
// 所以如果你愿意，也可以尝试用多线程加速训练过程

/**
 * 怎么加速PCFG训练过程？据助教所知，没有公开文献提出过有效的加速方法（因为这么做基本无学术价值）
 *
 * 但是统计学模型好就好在其数据是可加的。例如，假如我把数据集拆分成4个部分，并行训练4个不同的模型。
 * 然后我可以直接将四个模型的统计数据进行简单加和，就得到了和串行训练相同的模型了。
 *
 * 说起来容易，做起来不一定容易，你可能会碰到一系列具体的工程问题。如果你决定加速训练过程，祝你好运！
 *
 */

// 合并另一个模型的数据到当前模型
void model::merge(const model &other)
{
    // 合并preterminals
    for (int i = 0; i < other.preterminals.size(); ++i)
    {
        int idx = FindPT(other.preterminals[i]);
        if (idx == -1)
        {
            preterminals.push_back(other.preterminals[i]);
            preterm_freq[preterminals.size() - 1] = other.preterm_freq.at(i);
        }
        else
        {
            preterm_freq[idx] += other.preterm_freq.at(i);
        }
    }
    total_preterm += other.total_preterm;

    // 合并letters
    for (int i = 0; i < other.letters.size(); ++i)
    {
        int idx = FindLetter(other.letters[i]);
        if (idx == -1)
        {
            letters.push_back(other.letters[i]);
            letters_freq[letters.size() - 1] = other.letters_freq.at(i);
        }
        else
        {
            letters_freq[idx] += other.letters_freq.at(i);
            // 合并segment values
            for (const auto &kv : other.letters[i].values)
            {
                string val = kv.first;
                int val_idx = kv.second;
                if (letters[idx].values.find(val) == letters[idx].values.end())
                {
                    int new_idx = letters[idx].values.size();
                    letters[idx].values[val] = new_idx;
                    letters[idx].freqs[new_idx] = other.letters[i].freqs.at(val_idx);
                }
                else
                {
                    int exist_idx = letters[idx].values[val];
                    letters[idx].freqs[exist_idx] += other.letters[i].freqs.at(val_idx);
                }
            }
        }
    }
    // 合并digits
    for (int i = 0; i < other.digits.size(); ++i)
    {
        int idx = FindDigit(other.digits[i]);
        if (idx == -1)
        {
            digits.push_back(other.digits[i]);
            digits_freq[digits.size() - 1] = other.digits_freq.at(i);
        }
        else
        {
            digits_freq[idx] += other.digits_freq.at(i);
            for (const auto &kv : other.digits[i].values)
            {
                string val = kv.first;
                int val_idx = kv.second;
                if (digits[idx].values.find(val) == digits[idx].values.end())
                {
                    int new_idx = digits[idx].values.size();
                    digits[idx].values[val] = new_idx;
                    digits[idx].freqs[new_idx] = other.digits[i].freqs.at(val_idx);
                }
                else
                {
                    int exist_idx = digits[idx].values[val];
                    digits[idx].freqs[exist_idx] += other.digits[i].freqs.at(val_idx);
                }
            }
        }
    }
    // 合并symbols
    for (int i = 0; i < other.symbols.size(); ++i)
    {
        int idx = FindSymbol(other.symbols[i]);
        if (idx == -1)
        {
            symbols.push_back(other.symbols[i]);
            symbols_freq[symbols.size() - 1] = other.symbols_freq.at(i);
        }
        else
        {
            symbols_freq[idx] += other.symbols_freq.at(i);
            for (const auto &kv : other.symbols[i].values)
            {
                string val = kv.first;
                int val_idx = kv.second;
                if (symbols[idx].values.find(val) == symbols[idx].values.end())
                {
                    int new_idx = symbols[idx].values.size();
                    symbols[idx].values[val] = new_idx;
                    symbols[idx].freqs[new_idx] = other.symbols[i].freqs.at(val_idx);
                }
                else
                {
                    int exist_idx = symbols[idx].values[val];
                    symbols[idx].freqs[exist_idx] += other.symbols[i].freqs.at(val_idx);
                }
            }
        }
    }
}

// 多线程训练辅助函数
void train_worker(model *mdl, const vector<string> &pwds)
{
    cout << "[Thread " << this_thread::get_id() << "] Start training, size=" << pwds.size() << endl;
    for (const string &pw : pwds)
    {
        mdl->parse(pw);
    }
    cout << "[Thread " << this_thread::get_id() << "] Finish training." << endl;
}

// 训练的wrapper，实际上就是读取训练集
void model::train(string path)
{
    // 1. 读取所有口令到内存
    vector<string> all_pwds;
    string pw;
    ifstream train_set(path);
    int lines = 0;
    cout << "Training..." << endl;
    cout << "Training phase 1: reading and parsing passwords..." << endl;
    while (train_set >> pw)
    {
        lines += 1;
        if (lines % 10000 == 0)
        {
            cout << "Lines processed: " << lines << endl;
            if (lines > 3000000)
            {
                break;
            }
        }
        all_pwds.push_back(pw);
    }
    cout << "[Main] Total passwords loaded: " << all_pwds.size() << endl;
    // 2. 分成4份
    size_t total = all_pwds.size();
    size_t part = (total + 3) / 4;
    vector<string> pwds1(all_pwds.begin(), all_pwds.begin() + min(part, total));
    vector<string> pwds2(all_pwds.begin() + min(part, total), all_pwds.begin() + min(2 * part, total));
    vector<string> pwds3(all_pwds.begin() + min(2 * part, total), all_pwds.begin() + min(3 * part, total));
    vector<string> pwds4(all_pwds.begin() + min(3 * part, total), all_pwds.end());

    cout << "[Main] pwds1 size: " << pwds1.size() << endl;
    cout << "[Main] pwds2 size: " << pwds2.size() << endl;
    cout << "[Main] pwds3 size: " << pwds3.size() << endl;
    cout << "[Main] pwds4 size: " << pwds4.size() << endl;

    // 3. 启动3个线程分别训练3个模型，主线程自己处理pwds1
    model m1, m2, m3, m4;
    cout << "[Main] Launching threads..." << endl;
    thread t2(train_worker, &m2, cref(pwds2));
    thread t3(train_worker, &m3, cref(pwds3));
    thread t4(train_worker, &m4, cref(pwds4));
    // 主线程自己处理pwds1
    train_worker(&m1, pwds1);
    t2.join();
    t3.join();
    t4.join();
    cout << "[Main] All threads joined." << endl;

    // 4. 合并4个模型
    cout << "[Main] Merging model 1..." << endl;
    this->merge(m1);
    cout << "[Main] Merging model 2..." << endl;
    this->merge(m2);
    cout << "[Main] Merging model 3..." << endl;
    this->merge(m3);
    cout << "[Main] Merging model 4..." << endl;
    this->merge(m4);
    cout << "[Main] Merge finished." << endl;
}

/// @brief 在模型中找到一个PT的统计数据
/// @param pt 需要查找的PT
/// @return 目标PT在模型中的对应下标
int model::FindPT(PT pt)
{
    for (int id = 0; id < preterminals.size(); id += 1)
    {
        if (preterminals[id].content.size() != pt.content.size())
        {
            continue;
        }
        else
        {
            bool equal_flag = true;
            for (int idx = 0; idx < preterminals[id].content.size(); idx += 1)
            {
                if (preterminals[id].content[idx].type != pt.content[idx].type || preterminals[id].content[idx].length != pt.content[idx].length)
                {
                    equal_flag = false;
                    break;
                }
            }
            if (equal_flag == true)
            {
                return id;
            }
        }
    }
    return -1;
}

/// @brief 在模型中找到一个letter segment的统计数据
/// @param seg 要找的letter segment
/// @return 目标letter segment的对应下标
int model::FindLetter(segment seg)
{
    for (int id = 0; id < letters.size(); id += 1)
    {
        if (letters[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

/// @brief 在模型中找到一个digit segment的统计数据
/// @param seg 要找的digit segment
/// @return 目标digit segment的对应下标
int model::FindDigit(segment seg)
{
    for (int id = 0; id < digits.size(); id += 1)
    {
        if (digits[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

int model::FindSymbol(segment seg)
{
    for (int id = 0; id < symbols.size(); id += 1)
    {
        if (symbols[id].length == seg.length)
        {
            return id;
        }
    }
    return -1;
}

void PT::insert(segment seg)
{
    content.emplace_back(seg);
}

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
    {
        freqs[values[value]] += 1;
    }
}

void segment::order()
{
    for (pair<string, int> value : values)
    {
        ordered_values.emplace_back(value.first);
    }
    // cout << "value size:" << ordered_values.size() << endl;
    std::sort(ordered_values.begin(), ordered_values.end(),
              [this](const std::string &a, const std::string &b)
              {
                  return freqs.at(values[a]) > freqs.at(values[b]);
              });

    // 将排序后的频率存入 ordered_freqs 并计算 total_freq
    for (const std::string &val : ordered_values)
    {
        ordered_freqs.emplace_back(freqs.at(values[val]));
        total_freq += freqs.at(values[val]);
    }
    // for (string val : ordered_values)
    // {
    //     ordered_freqs.emplace_back(freqs.at(values[val]));
    //     total_freq += freqs.at(values[val]);
    // }
}

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0; // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    // 请学会使用这种方式写for循环：for (auto it : iterable)
    // 相信我，以后你会用上的。You're welcome :)
    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits[id].insert(curr_part);
                        digits_freq[id] = 1;
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 1;
            curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 2;
            curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits_freq[id] = 1;
                        digits[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 3;
            curr_part += ch;
        }
    }
    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            {
                int id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq[id] = 1;
                letters[id].insert(curr_part);
            }
            else
            {
                int id = FindLetter(seg);
                letters_freq[id] += 1;
                letters[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            {
                int id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq[id] = 1;
                digits[id].insert(curr_part);
            }
            else
            {
                int id = FindDigit(seg);
                digits_freq[id] += 1;
                digits[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            {
                int id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq[id] = 1;
                symbols[id].insert(curr_part);
            }
            else
            {
                int id = FindSymbol(seg);
                symbols_freq[id] += 1;
                symbols[id].insert(curr_part);
            }
            curr_part.clear();
            pt.insert(seg);
        }
    }
    // pt.PrintPT();
    // cout<<endl;
    // cout << FindPT(pt) << endl;
    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (int i = 0; i < pt.content.size(); i += 1)
        {
            pt.curr_indices.emplace_back(0);
        }
        int id = GetNextPretermID();
        // cout << id << endl;
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        int id = FindPT(pt);
        // cout << id << endl;
        preterm_freq[id] += 1;
    }
}

void segment::PrintSeg()
{
    if (type == 1)
    {
        cout << "L" << length;
    }
    if (type == 2)
    {
        cout << "D" << length;
    }
    if (type == 3)
    {
        cout << "S" << length;
    }
}

void segment::PrintValues()
{
    // order();
    for (string iter : ordered_values)
    {
        cout << iter << " freq:" << freqs[values[iter]] << endl;
    }
}

void PT::PrintPT()
{
    for (auto iter : content)
    {
        iter.PrintSeg();
    }
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < preterminals.size(); i += 1)
    {
        preterminals[i].PrintPT();
        // cout << preterminals[i].curr_indices.size() << endl;
        cout << " freq:" << preterm_freq[i];
        cout << endl;
    }
    // order();
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)];
        cout << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        letters[i].PrintSeg();
        // letters[i].PrintValues();
        cout << " freq:" << letters_freq[i];
        cout << endl;
    }
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].PrintSeg();
        // digits[i].PrintValues();
        cout << " freq:" << digits_freq[i];
        cout << endl;
    }
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].PrintSeg();
        // symbols[i].PrintValues();
        cout << " freq:" << symbols_freq[i];
        cout << endl;
    }
}

bool compareByPretermProb(const PT &a, const PT &b)
{
    return a.preterm_prob > b.preterm_prob; // 降序排序
}

void model::order()
{
    cout << "Training phase 2: Ordering segment values and PTs..." << endl;
    for (PT pt : preterminals)
    {
        pt.preterm_prob = float(preterm_freq[FindPT(pt)]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    bool swapped;
    cout << "total pts" << ordered_pts.size() << endl;
    std::sort(ordered_pts.begin(), ordered_pts.end(), compareByPretermProb);
    cout << "Ordering letters" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < letters.size(); i += 1)
    {
        // cout << i << endl;
        letters[i].order();
    }
    cout << "Ordering digits" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < digits.size(); i += 1)
    {
        digits[i].order();
    }
    cout << "ordering symbols" << endl;
    // cout << "total letters" << endl;
    for (int i = 0; i < symbols.size(); i += 1)
    {
        symbols[i].order();
    }
}