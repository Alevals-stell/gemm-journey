Device Name: Tesla T4
Memory Clock Rate: 5001000 kHz
Memory Bus Width: 256 bits
Theoretical Bandwidth: 320.064 GB/s

H2D_time: 0.174
D2H_time: 0.450
time: 0.005
gflops: 21.970GFLOPS
bandwidth: 263.639GB/s

thread = 64/128/256/512/1024都是接近下面这个结果
PASS
time: 0.005
gflops: 22.010GFLOPS
bandwidth: 264.121GB/s

thread = 32
H2D_time: 0.174
D2H_time: 0.281
PASS
time: 0.006
gflops: 15.414GFLOPS
bandwidth: 184.967GB/s



实验 1 和 2：DRAM 实际搬运量
DRAM 读	DRAM 写	合计	逻辑需要	实际/逻辑
SoA scalar	866.8 MB	449.8 MB	1.317 GB	读 800 + 写 400 = 1.2 GB	1.10
AoS	1.31 GB	1.33 GB	2.64 GB	同上 1.2 GB	2.20
AoS 为什么读写都是 1.2 GB：读的时候 c 和 a、b 在同一个缓存扇区里，整个数组都被读进来；写的时候每个扇区都被改过一部分，整个扇区都要写回。读写各多搬了约 0.4 到 0.8 GB 的无用数据。

访存受限 kernel：耗时 ≈ DRAM 实际搬运字节数 ÷ 带宽。 优化的唯一方向是少搬数据。
有效带宽和 ncu 的 DRAM 利用率口径不同，要看 profiler 给出的字节数，而不是自己推算。
性能受"在途字节数"约束，占用率、向量化、每线程处理量都是提高它的不同手段，互相可以替代。
GPU 偏好 SoA：相邻线程访问相邻地址，不要把同一线程的数据放一起。
提频不提速是判断访存受限的一个简单实验。
工具的提示要结合瓶颈判断，上次 ncu 建议提高占用率，在 DRAM 已饱和的情况下没有意义。