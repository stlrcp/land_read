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
#ifndef __SAMPLE_MEMCPYH2D2H_H__
#define __SAMPLE_MEMCPYH2D2H_H__
#include <iostream>
#include <getopt.h>
#include <vector>
// #include "acl/acl.h"
#include "aicl_adaptor_iluvatar.cpp"
#include "aicl.h"
#define SAMPLE_PRT(fmt...)   \
    do { \
        printf("[%s]-%d: ", __FUNCTION__, __LINE__); \
        printf(fmt); \
    } while (0)

int SUCCESS = 0;
int FAILED = 1;

uint32_t g_release_cycle    = -1;
uint32_t g_number_of_cycles = 1;
uint32_t g_device_id        = 0;
size_t g_memory_size        = 10485760;
uint32_t g_write_back_host  = 1;
uint32_t g_memory_reuse     = 1;
aiclrtContext g_context      = nullptr;
aiclrtRunMode g_run_mode     = AICL_DEVICE;

#endif /* End of #ifndef __SAMPLE_MEMCPYH2D2H_H__ */