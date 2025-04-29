# Lab2口令猜测SIMD编程
上述文件中，main.cpp文件没有改动，md5.h增加了那9个函数的并行优化后的函数;
编译指令如下：
```
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O1
g++ main.cpp train.cpp guessing.cpp md5.cpp -o main -O2
```

"NEON指令集实现SIMD"文件夹中main_simd.cpp与md5.cpp使用NEON指令进行SIMD编程;
编程指令如下:
```
g++ main_simd.cpp train.cpp guessing.cpp md5_simd.cpp -o main
g++ main_simd.cpp train.cpp guessing.cpp md5_simd.cpp -o main -O1
g++ main_simd.cpp train.cpp guessing.cpp md5_simd.cpp -o main -O2
```

SSE指令集文件中记录了我在x86环境下使用SSE指令集实现串行和SIMD编程的代码,我对串行和并行代码使用VTuen进行了profiling,截图在"profiling截屏"文件夹中

"不使用任何指令集实现SIMD"文件夹中main_noNEON.cpp,md5_noNEON.cpp和md5_noNEON.h在不使用NEON指令的条件下实现SIMD编程;
编程指令如下:
```
g++ main_noNEON.cpp train.cpp guessing.cpp md5_noNEON.cpp -o main
g++ main_noNEON.cpp train.cpp guessing.cpp md5_noNEON.cpp -o main -O1
g++ main_noNEON.cpp train.cpp guessing.cpp md5_noNEON.cpp -o main -O2
```

"控制“单指令多数据”中的数据量"文件夹中main_batchsize2_4_8.cpp,md5_batchsize2_4_8.cpp分别实现控制“单指令多数据”中的数据量为2,4,8;
编程指令如下:
```
g++ main_batchsize2_4_8.cpp train.cpp guessing.cpp md5_batchsize2_4_8.cpp -o main
g++ main_batchsize2_4_8.cpp train.cpp guessing.cpp md5_batchsize2_4_8.cpp -o main -O1
g++ main_batchsize2_4_8.cpp train.cpp guessing.cpp md5_batchsize2_4_8.cpp -o main -O2
```

以上代码均使用不同的测试样例验证了其正确性