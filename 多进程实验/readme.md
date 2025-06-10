# 多进程并行化实验
在本次多进程并行化的实验中，我实现了两种多进程的方式，分别命名为版本一与版本二，详细介绍在报告中给出

### 版本一编译指令
```
mpic++ -o main guessing_MPI.cpp train.cpp main_MPI.cpp md5.cpp
```

### 版本二编译指令
```
mpic++ -o main guessing_MPI_pro.cpp train.cpp main_MPI_pro.cpp md5.cpp
```