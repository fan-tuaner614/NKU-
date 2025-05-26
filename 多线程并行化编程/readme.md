# Lab3多线程并行化编程
上述代码中，

guessing._openmp.cpp是使用openmp并行化编程的代码

guessing_pthread_1.cpp是使用pthread并行化编程的代码的方法1
guessing_pthread_2.cpp是使用pthread并行化编程的代码的方法2
guessing_pthread_3.cpp是使用pthread并行化编程的代码的方法3

guessing_pthread_pro.cpp是使用pthread并行化编程的代码的最终优化代码

correctness_guess_2.cpp是用来测试代码guessing_pthread_2.cpp正确性的代码

其编译编译如下：
```
g++ guessing_pthread_2.cpp train.cpp correctness_guess_2.cpp md5.cpp -o main
g++ guessing_pthread_2.cpp train.cpp correctness_guess_2.cpp md5.cpp -o main -O1
g++ guessing_pthread_2.cpp train.cpp correctness_guess_2.cpp md5.cpp -o main -O2
```

correctness_guess.cpp是用来测试代码guessing_pthread_1.cpp、guessing_pthread_3.cpp、guessing._openmp.cpp正确性的代码

其编译编译如下：
```
g++ guessing.cpp train.cpp correctness_guess.cpp md5.cpp -o main
g++ guessing.cpp train.cpp correctness_guess.cpp md5.cpp -o main -O1
g++ guessing.cpp train.cpp correctness_guess.cpp md5.cpp -o main -O2

g++ guessing_pthread_1.cpp train.cpp correctness_guess.cpp md5.cpp -o main
g++ guessing_pthread_1.cpp train.cpp correctness_guess.cpp md5.cpp -o main -O1
g++ guessing_pthread_1.cpp train.cpp correctness_guess.cpp md5.cpp -o main -O2

g++ guessing_pthread_3.cpp train.cpp correctness_guess.cpp md5.cpp -o main
g++ guessing_pthread_3.cpp train.cpp correctness_guess.cpp md5.cpp -o main -O1
g++ guessing_pthread_3.cpp train.cpp correctness_guess.cpp md5.cpp -o main -O2

g++ guessing_openmp.cpp train.cpp correctness_guess.cpp md5.cpp -o main -fopenmp
g++ guessing_openmp.cpp train.cpp correctness_guess.cpp md5.cpp -o main -fopenmp -O1
g++ guessing_openmp.cpp train.cpp correctness_guess.cpp md5.cpp -o main -fopenmp -O2
```

correctness_guess_pro.cpp是用来测试代码guessing_pthread_pro.cpp正确性的代码

其编译编译如下：
```
g++ guessing_pthread_pro.cpp train.cpp correctness_guess_pro.cpp md5.cpp -o main
g++ guessing_pthread_pro.cpp train.cpp correctness_guess_pro.cpp md5.cpp -o main -O1
g++ guessing_pthread_pro.cpp train.cpp correctness_guess_pro.cpp md5.cpp -o main -O2
```