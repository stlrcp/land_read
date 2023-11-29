中文(./README.md)
# 内存拷贝（Host&Device）
调用AICL的内存管理&数据传输相关接口完成Host与Device间的数据传输。

#### 场景说明
功能：提供Host与Device间数据传输样例，帮助开发者理解相关接口与调用流程。    
输入：控制参数（是否内存复用，单次申请内存大小等）。    
输出：打屏输出device侧内存占用量及每次搬运后的内存数据。  

#### 样例运行
切换到样例目录，执行如下命令：
```
cd scripts
bash sample_build.sh
cd ../build/src
./main --release_cycle 10 --number_of_cycles 1 --device_id 0 --memory_size 10485760 --memory_reuse 0 --write_back_host 0
```
-  **--release_cycle RELEASE_CYCLE**          ：释放周期，每执行此参数数量次内存拷贝后进行一次释放，-1代表死循环
-  **--number_of_cycles NUMBER_OF_CYCLES**    ：循环周期，释放周期循环执行次数，-1代表死循环
-  **--device_id DEVICE_ID**                  ：设备ID
-  **--memory_size MEMORY_SIZE**              ：单次申请内存块大小，单位为字节
-  **--memory_reuse MEMORY_REUSE**            ：内存是否进行复用，1表进行内存复用，0表不进行内存复用
-  **--write_back_host WRITE_BACK_HOST**      ：是否要回传Host，1表回传Host，0表不回传Host

**注：由于内存不复用且一致不释放时，会在某一时刻将device内存占满，此时会循环打印内存申请失败，直至系统自动将程序杀死。这是为了体现device内存的容量上限以及让用户了解内存释放的机制，并非BUG。**

#### 查看结果
命令行可以查看每次内存拷贝前、内存拷贝后、内存回传后的内存取值及device内存状况。
