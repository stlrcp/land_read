/**
 *  Copyright © 2023 Iluvatar CoreX. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 */
#include "memcpy_host_device.h"
#include <unistd.h>
using namespace std;
// 检查入参
void memcpyh2d_usage(char *sPrgNm)
{
    aiclRet Ret = aiclrtGetRunMode(&g_run_mode);
    if (Ret == AICL_RET_SUCCESS) {
        if (g_run_mode == AICL_HOST) {
            SAMPLE_PRT(" Running in Host!\n");
        } else if (g_run_mode == AICL_DEVICE) {
            SAMPLE_PRT(" Running in Device!\n");
        } else {
            SAMPLE_PRT(" Running in Invalid platform! runMode:%u\n", g_run_mode);
            return;
        }
    } else {
        SAMPLE_PRT(" Get run mode fail! aicl ret:%#x\n", Ret);
        return;
    }

    SAMPLE_PRT("\n/*********************************************************/\n");
    SAMPLE_PRT("Usage :\n");
    SAMPLE_PRT("\t example: ./main --release_cycle -1\
    --number_of_cycles 1 --device_id 0 --memory_size 10485760\
    --write_back_host 0 --memory_reuse 1\n");
    SAMPLE_PRT("\t\n");
    SAMPLE_PRT("\t  --release_cycle: Specify the release period.\n");
    SAMPLE_PRT("\t  --number_of_cycles: The number of overall process cycles.\n");
    SAMPLE_PRT("\t  --device_id: The ID of device currently in use.\n");
    SAMPLE_PRT("\t  --memory_size: Single request memory size, The units are Byte\n");
    SAMPLE_PRT("\t  --memory_reuse: Whether to reuse memory\n");
    SAMPLE_PRT("\t  --write_back_host: Whether to send back the host\n");
    SAMPLE_PRT("\n/*********************************************************/\n\n");
}

// 获取参数
int32_t get_option(int32_t argc, char **argv)
{
    int32_t c = 0;
    int32_t option_index = 0;
    int32_t ret = 0;
    struct option long_options[] =
    {
        {"release_cycle",       1, nullptr, 'r'},
        {"number_of_cycles",    1, nullptr, 'n'},
        {"device_id",           1, nullptr, 'd'},
        {"memory_size",         1, nullptr, 's'},
        {"write_back_host",     1, nullptr, 'w'},
        {"memory_reuse",        1, nullptr, 'm'},
        {nullptr,               0, nullptr, 0}
    };

    while (1) {
        option_index = 0;
        c = getopt_long(argc, argv, "r:n:d:s:", long_options, &option_index);
        if (c == -1) {
            break;
        }
        switch (c) {
            case 'r':
                g_release_cycle = atoi(optarg);
                break;
            case 'n':
                g_number_of_cycles = atoi(optarg);
                break;
            case 'd':
                g_device_id = atoi(optarg);
                break;
            case 's':
                g_memory_size = atoi(optarg);
                break;
            case 'w':
                g_write_back_host = atoi(optarg);
                break;
            case 'm':
                g_memory_reuse = atoi(optarg);
                break;
            default:
                SAMPLE_PRT("unsupport option!\n");
                break;
        }
    }

    aiclRet Ret = aiclrtGetRunMode(&g_run_mode);
    if (Ret == AICL_RET_SUCCESS) {
        if (g_run_mode == AICL_HOST) {
            SAMPLE_PRT(" Running in Host!\n");
        } else if (g_run_mode == AICL_DEVICE) {
            SAMPLE_PRT(" Running in Device!\n");
        } else {
            SAMPLE_PRT(" Running in Invalid platform! runMode:%u\n", g_run_mode);
            return FAILED;
        }
    } else {
        SAMPLE_PRT(" Get run mode fail! aicl ret:%#x\n", Ret);
        return FAILED;
    }
    SAMPLE_PRT("\n/*********************************************************/\n");
    SAMPLE_PRT("\nUsing params are as follows.\n");
    SAMPLE_PRT("g_release_cycle: %u \n", g_release_cycle);
    SAMPLE_PRT("g_number_of_cycles: %u \n", g_number_of_cycles);
    SAMPLE_PRT("g_device_id: %u \n", g_device_id);
    SAMPLE_PRT("g_memory_size: %zu \n", g_memory_size);
    SAMPLE_PRT("g_write_back_host: %u \n", g_write_back_host);
    SAMPLE_PRT("g_memory_reuse: %u \n", g_memory_reuse);
    SAMPLE_PRT("\n/*********************************************************/\n");
    return SUCCESS;
}

int32_t setup_aicl_device()
{
    aiclRet Ret = aiclInit(nullptr);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aiclInit fail with %d.\n", Ret);
        return Ret;
    }
    SAMPLE_PRT("aiclInit succ.\n");

    Ret = aiclrtSetDevice(g_device_id);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aiclrtSetDevice %u fail with %d.\n", g_device_id, Ret);
        aiclFinalize();
        return Ret;
    }
    SAMPLE_PRT("aiclrtSetDevice(%u) succ.\n", g_device_id);

    Ret = aiclrtCreateContext(&g_context, g_device_id);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aicl create context failed with %d.\n", Ret);
        aiclrtResetDevice(g_device_id);
        aiclFinalize();
        return Ret;
    }
    SAMPLE_PRT("create context success\n");

    Ret = aiclrtGetCurrentContext(&g_context);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("get current context failed\n");
        aiclrtDestroyContext(g_context);
        g_context = nullptr;
        aiclrtResetDevice(g_device_id);
        aiclFinalize();
        return Ret;
    }
    SAMPLE_PRT("get current context success\n");

    return SUCCESS;
}

void print_data(string message, void *data, size_t size, aiclrtMemcpyKind kind)
{
    void* host_data = nullptr;
    aiclRet Ret = aiclrtMallocHost(&host_data, size);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aiclrtMallocHost failed\n");
        return;
    }
    Ret = aiclrtMemcpy(host_data, size, data, size, kind);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aiclrtMemcpy failed\n");
        return;
    }
    SAMPLE_PRT("\n%s is %x\n", message.c_str(), *(unsigned char*)host_data);
    Ret = aiclrtFreeHost(host_data);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aiclrtFreeHost failed\n");
	return;
    }
    return;
}

int32_t search_memory(uint32_t cycles_time, uint32_t release_cycle_time)
{
    size_t free = 0;
    size_t total = 0;
    aiclRet Ret = aiclrtGetMemInfo(AICL_DDR_MEM, &free, &total);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aiclrtGetMemInfo failed\n");
        return Ret;
    }
    SAMPLE_PRT("At number_of_cycles = %u, release_cycle = %u, HBM free memory:%zu Byte,   \
               HBM total memory:%zu Byte.\n",cycles_time, release_cycle_time, free, total);
    Ret = aiclrtGetMemInfo(AICL_HBM_MEM, &free, &total);
    if (Ret != AICL_RET_SUCCESS) {
        SAMPLE_PRT("aiclrtGetMemInfo failed\n");
        return Ret;
    }
    SAMPLE_PRT("At number_of_cycles = %u, release_cycle = %u, HBM free memory:%zu Byte,   \
               HBM total memory:%zu Byte.\n",cycles_time, release_cycle_time, free, total);
    return SUCCESS;
}

void destroy_aicl_device()
{
    if (g_context) {
        aiclrtDestroyContext(g_context);
        g_context = nullptr;
        aiclrtResetDevice(g_device_id);
        aiclFinalize();
    }
}

int32_t main(int32_t argc, char *argv[])
{
    int ret = SUCCESS;
    // 检查参数个数
    if (argc < 4) {
        SAMPLE_PRT("\nInput parameter's num:%d is not enough!\n", argc);
        memcpyh2d_usage(argv[0]);
        return FAILED;
    }
    // 获取入参
    ret = get_option(argc, &(*argv));
    if (ret != SUCCESS) {
        SAMPLE_PRT("get_option failed!\n");
        return FAILED;
    }
    // aicl资源初始化
    ret = setup_aicl_device();
    if (ret != SUCCESS) {
        SAMPLE_PRT("Setup Device failed! ret code:%#x\n", ret);
        return FAILED;
    }
    // 内存使用前，首次查询Device内存
    ret = search_memory(0, 0);
    if (ret != SUCCESS) {
        SAMPLE_PRT("search_memory failed! ret code:%#x\n", ret);
        return FAILED;
    }
    // 根据循环次数设置循环
    uint32_t cycles_time = 0;
    aiclRet Ret = AICL_RET_SUCCESS;
    while (g_number_of_cycles - cycles_time != 0)
    {
        // 设置内存list，方便后续释放
        void* host_buffer = nullptr;
        void* device_buffer = nullptr;
        vector<void*> host_mem_list;
        vector<void*> dev_mem_list;
        // 如果内存复用，则需要在释放周期循环外申请内存
        if (g_memory_reuse) {
            // 申请host内存
            Ret = aiclrtMallocHost(&host_buffer, g_memory_size);
            if (Ret != AICL_RET_SUCCESS) {
                SAMPLE_PRT("aiclrtMallocHost failed\n");
                continue;
            }
            // 申请device内存
            Ret = aiclrtMalloc(&device_buffer, g_memory_size, AICL_MEM_MALLOC_HUGE_FIRST);
            if (Ret != AICL_RET_SUCCESS) {
                SAMPLE_PRT("aiclrtMalloc failed\n");
                continue;
            }
            host_mem_list.push_back(host_buffer);
            dev_mem_list.push_back(device_buffer);
        }
        // 根据释放周期设置循环
        uint32_t release_cycle_time = 0;
        while (g_release_cycle - release_cycle_time != 0)
        {
            // 如果内存不复用，则需要在释放周期循环内申请内存
            if (!g_memory_reuse) {
                // 申请host内存
                Ret = aiclrtMallocHost(&host_buffer, g_memory_size);
                if (Ret != AICL_RET_SUCCESS) {
                    SAMPLE_PRT("aiclrtMallocHost failed\n");
                    continue;
                }
                // 申请device内存
                Ret = aiclrtMalloc(&device_buffer, g_memory_size, AICL_MEM_MALLOC_HUGE_FIRST);
                if (Ret != AICL_RET_SUCCESS) {
                    SAMPLE_PRT("aiclrtMalloc failed\n");
                    continue;
                }
                host_mem_list.push_back(host_buffer);
                dev_mem_list.push_back(device_buffer);
            }
            // 初始化内存，构建全7的数据
            Ret = aiclrtMemset (host_buffer, g_memory_size, 7, g_memory_size);
            if (Ret != AICL_RET_SUCCESS) {
                SAMPLE_PRT("aiclrtMemset failed\n");
                continue;
            }
            // 查看初始化后的内存
            print_data("At first memset host data", host_buffer, g_memory_size, AICL_MEMCPY_HOST_TO_HOST);
            // 内存拷贝H2D
            Ret = aiclrtMemcpy(device_buffer, g_memory_size, host_buffer, g_memory_size, AICL_MEMCPY_HOST_TO_DEVICE);
            if (Ret != AICL_RET_SUCCESS) {
                SAMPLE_PRT("aiclrtMemcpy failed\n");
                continue;
            }
            // 查看device上的数据
            print_data("At memcpy device data", device_buffer, g_memory_size, AICL_MEMCPY_DEVICE_TO_HOST);
            // 判断是否要回传Host
            if (g_write_back_host) {
                // 申请回传所需的Host内存
                void* write_back_buffer = nullptr;
                Ret = aiclrtMallocHost(&write_back_buffer, g_memory_size);
                if (Ret != AICL_RET_SUCCESS) {
                    SAMPLE_PRT("aiclrtMallocHost failed\n");
                    continue;
                }
                // 内存拷贝D2H
                Ret = aiclrtMemcpy(write_back_buffer, g_memory_size, device_buffer, g_memory_size, AICL_MEMCPY_DEVICE_TO_HOST);
                if (Ret != AICL_RET_SUCCESS) {
                    SAMPLE_PRT("aiclrtMemcpy failed\n");
                    continue;
                }
                // 查看回传后的内存
                print_data("At write back memcpy host data", write_back_buffer, g_memory_size, AICL_MEMCPY_HOST_TO_HOST);
                // 回传所申请的Host内存单独释放
                aiclRet Ret = aiclrtFreeHost(write_back_buffer);
                if (Ret != AICL_RET_SUCCESS) {
                    SAMPLE_PRT("aiclrtFreeHost failed\n");
                    continue;
                }
            }
            release_cycle_time += 1;
            ret = search_memory(cycles_time, release_cycle_time);
            if (ret != SUCCESS) {
                SAMPLE_PRT("search_memory failed! ret code:%#x\n", ret);
                continue;
            }
        }
        // 一个释放周期循环完毕后，进行内存释放
        for (auto host_buffer : host_mem_list) {
            aiclRet Ret = aiclrtFreeHost(host_buffer);
            if (Ret != AICL_RET_SUCCESS) {
                SAMPLE_PRT("aiclrtFreeHost failed\n");
                continue;
            }
        }
        for (auto device_buffer : dev_mem_list) {
            aiclRet Ret = aiclrtFree(device_buffer);
            if (Ret != AICL_RET_SUCCESS) {
                SAMPLE_PRT("aiclrtFree failed\n");
                continue;
            }
        }
        // 调用内存查询接口，查询device内存
        cycles_time += 1;
        ret = search_memory(cycles_time, release_cycle_time);
        if (ret != SUCCESS) {
            SAMPLE_PRT("search_memory failed! ret code:%#x\n", ret);
            return FAILED;
        }
    }
    // 释放aicl资源
    destroy_aicl_device();
    SAMPLE_PRT("run success!\n");
    return SUCCESS;
}
