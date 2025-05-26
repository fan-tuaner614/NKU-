#ifndef PCFG_H
#define PCFG_H

#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <pthread.h> // 添加线程相关头文件
#include <set>

using namespace std;

class segment
{
public:
    int type;   // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符
    int length; // 长度，例如S6的长度就是6
    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values;
    unordered_map<int, int> freqs;

    segment(int type, int length) : type(type), length(length) {}
    void insert(string value);
    void order();
    void PrintSeg();
    void PrintValues();
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
    vector<PT> preterminals;
    vector<segment> letters, digits, symbols;
    unordered_map<int, int> preterm_freq, letters_freq, digits_freq, symbols_freq;
    vector<PT> ordered_pts;

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
};

class ThreadPool; // Forward declaration

class PriorityQueue
{
public:
    vector<PT> priority;
    model m;
    vector<string> guesses;
    int total_guesses = 0;
    ThreadPool *pool; // Global thread pool as member variable

    // PriorityQueue();  // Constructor declaration
    // ~PriorityQueue(); // Destructor declaration
    void PopNextParallel();
    void CalProb(PT &pt);
    void init();
    void Generate(PT pt);
    void Generate(PT pt, std::vector<std::string> *guesses);
    void PopNext();
};

#endif