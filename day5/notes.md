这个实验用的是google colab T4. 每个block最多1024个线程。
1、threads > 1024会导致启动失败，提示invalid argument
2、当没有设置if (i < n)时，发生线程越界访问数组地址，运行报错：

Invalid __global__ read of size 4 bytes
=========     at vec_add(const float *, const float *, float *, int)+0x70
=========     by thread (707,0,0) in block (976,0,0)
=========     Access to 0x7fafa53d0b0c is out of bounds
=========     and is 525 bytes after the nearest allocation at 0x7fafa5000000 of size 4,000,000 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x8fd3] in vec_add
========= 
CUDA error: unspecified launch failure at vec_add.cu:44
========= Target application returned an error
========= ERROR SUMMARY: 449 errors
========= ERROR SUMMARY: 349 errors were not printed. Use --print-limit option to adjust the number of printed errors

这个实验中threads = 1024，总的线程数量是1,000,000，那么block = 1,000,000+1023 /1024 = 977。一共977个block，因为976*1024 = 999,424,
999,424 + 576 = 1,000,000。所以block976的576-1023线程全部越界，
越界的线程总数是977*1024-1,000,000 = 448。正好对应报错的信息。


实验3:
n / threads（不向上取整）报错：
========= COMPUTE-SANITIZER
Mismatch at 999936: gpu=0.000000, cpu=2499840.000000
Mismatch at 999937: gpu=0.000000, cpu=2499842.500000
Mismatch at 999938: gpu=0.000000, cpu=2499845.000000
Mismatch at 999939: gpu=0.000000, cpu=2499847.500000
Mismatch at 999940: gpu=0.000000, cpu=2499850.000000
FAIL (64 errors)
========= ERROR SUMMARY: 0 errors 

漏算的个数 = n % threads（1,000,000 % 256 = 64）