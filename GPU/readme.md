### GPU编程
在我的代码中guessing_1.cu是GPU编程的第一版代码，guessing_2.cu是GPU编程的第二版代码。

对于guessing_1.cu的编译指令如下：
```
nvcc correctness_guess.cpp train.cpp guessing_1.cu md5.cpp -lcudart -o test
nvcc correctness_guess.cpp train.cpp guessing_1.cu md5.cpp -lcudart -o test -O1
nvcc correctness_guess.cpp train.cpp guessing_1.cu md5.cpp -lcudart -o test -O2
```

对于guessing_2.cu的编译指令如下：
```
nvcc correctness_guess.cpp train.cpp guessing_2.cu md5.cpp -lcudart -o test
nvcc correctness_guess.cpp train.cpp guessing_2.cu md5.cpp -lcudart -o test -O1
nvcc correctness_guess.cpp train.cpp guessing_2.cu md5.cpp -lcudart -o test -O2
```