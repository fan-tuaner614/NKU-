#include "PCFG.h"
#include <mpi.h>
#include <string>
#include <vector>
using namespace std;

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices)
    {
        const auto &seg = pt.content[index];
        if (seg.type == 1)
        {
            int seg_idx = m.FindLetter(seg);
            if (m.letters[seg_idx].total_freq > 0)
            {
                pt.prob *= static_cast<float>(m.letters[seg_idx].ordered_freqs[idx]) / m.letters[seg_idx].total_freq;
            }
        }
        else if (seg.type == 2)
        {
            int seg_idx = m.FindDigit(seg);
            if (m.digits[seg_idx].total_freq > 0)
            {
                pt.prob *= static_cast<float>(m.digits[seg_idx].ordered_freqs[idx]) / m.digits[seg_idx].total_freq;
            }
        }
        else if (seg.type == 3)
        {
            int seg_idx = m.FindSymbol(seg);
            if (m.symbols[seg_idx].total_freq > 0)
            {
                pt.prob *= static_cast<float>(m.symbols[seg_idx].ordered_freqs[idx]) / m.symbols[seg_idx].total_freq;
            }
        }
        index++;
    }
}

void PriorityQueue::init()
{
    for (const PT &pt_template : m.ordered_pts)
    {
        PT pt = pt_template;
        pt.max_indices.clear();
        for (const segment &seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            else if (seg.type == 2)
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            else if (seg.type == 3)
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        if (m.total_preterm > 0)
        {
            pt.preterm_prob = static_cast<float>(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        }
        else
        {
            pt.preterm_prob = 0.0f;
        }
        CalProb(pt);
        priority.push(pt);
    }
}

void PriorityQueue::Generate(const PT &pt, vector<string> &local_guesses)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_guesses.clear();

    if (pt.content.size() == 1)
    {
        segment *a = nullptr;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else if (pt.content[0].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        if (a && !pt.max_indices.empty())
        {
            int total_vals = pt.max_indices[0];
            local_guesses.reserve(total_vals / size + 1);
            for (int i = rank; i < total_vals; i += size)
            {
                local_guesses.emplace_back(a->ordered_values[i]);
            }
        }
    }
    else
    {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            const auto &seg = pt.content[seg_idx];
            if (seg.type == 1)
                prefix += m.letters[m.FindLetter(seg)].ordered_values[idx];
            else if (seg.type == 2)
                prefix += m.digits[m.FindDigit(seg)].ordered_values[idx];
            else if (seg.type == 3)
                prefix += m.symbols[m.FindSymbol(seg)].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1)
                break;
        }

        segment *a = nullptr;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)
            a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)
            a = &m.digits[m.FindDigit(pt.content[last])];
        else if (pt.content[last].type == 3)
            a = &m.symbols[m.FindSymbol(pt.content[last])];

        if (a && !pt.max_indices.empty())
        {
            int total_vals = pt.max_indices[last];
            local_guesses.reserve(total_vals / size + 1);
            for (int i = rank; i < total_vals; i += size)
            {
                local_guesses.emplace_back(prefix + a->ordered_values[i]);
            }
        }
    }
}

void PriorityQueue::PopNext()
{
    if (priority.empty())
        return;
    PT top_pt = priority.top();
    priority.pop();
    vector<string> local_guesses;
    Generate(top_pt, local_guesses);
    guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
    total_guesses += local_guesses.size();

    PT pt_to_expand = top_pt;
    vector<PT> new_pts = pt_to_expand.NewPTs();
    for (PT &new_pt : new_pts)
    {
        CalProb(new_pt);
        priority.push(move(new_pt));
    }
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