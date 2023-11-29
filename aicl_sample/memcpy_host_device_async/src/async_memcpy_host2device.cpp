/*
异步正常拷贝数据 原长度和目标长度相同,展示4种情况:H2H\H2D\D2D\D2H  检查是否报错
*/

#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <pthread.h>
#include <stdint.h>
#include <map>
#include <iostream>
#include <signal.h>
#include "aicl_adaptor_iluvatar.cpp"
#include "aicl.h"

// 需要使用到静态公共库的函数名

// 全局资源
aiclrtContext context = nullptr;						// 全局使用的ctx
aiclRet ret = 0;									// 定义主函数的返回状态
int32_t device_id = 0;								// 压力所在的deviceid
aiclrtStream stream;									// 异步拷贝使用的stream

// 拷贝的输入输出buff
void* src_buffer = nullptr;						// device端输入内存地址
void* dst_buffer = nullptr;						// device端输出内存地址
size_t copy_len = 1024*1024*1024;				// 展示1M数据的拷贝

aiclRet init_fromwork()
{
	aiclRet ret = 0;
	// 初始化aicl框架 set device和创建ctx
	ret = aiclInit("./aicl.json");					if (ret != AICL_RET_SUCCESS){printf("aicl init failed %d\n",ret);return -1;}					else{printf("acl init success \n");}
	ret = aiclrtSetDevice(device_id);				if(ret!=AICL_RET_SUCCESS){printf("aiclrtSetDevice fail! %d \n",ret);return -1;}				else{printf("set device success \n");}
	ret = aiclrtCreateContext(&context, device_id);	if(ret!=AICL_RET_SUCCESS){printf("create ctx fail! %d \n",ret);return -1;}					else{printf("create context success \n");}
	ret = aiclrtSetCurrentContext(context);			if(ret!=AICL_RET_SUCCESS){printf("set thread to current context fail! \n");return -1;}		else{printf("set contex success \n");}
	ret = aiclrtCreateStream(&stream);				if(ret!=AICL_RET_SUCCESS){printf("create stream fail! \n");return -1;}						else{printf("create stream success \n");}
	
	// 申请主机和device的内存
	ret = aiclrtMallocHost(&src_buffer, copy_len);	if(ret!=AICL_RET_SUCCESS){printf("rt malloc host fail! \n");return -1;}						else{printf("malloc host success \n");}
	ret = aiclrtMalloc(&dst_buffer, copy_len,AICL_MEM_MALLOC_HUGE_FIRST);	    
													if(ret!=AICL_RET_SUCCESS){printf("rt malloc device fail! \n");return -1;}						else{printf("malloc device success \n");}
	return AICL_RET_SUCCESS;
}

aiclRet uninit_formwork()
{
	//释放主机上的内存
	if(src_buffer != nullptr){aiclrtFreeHost(src_buffer);}
	if(dst_buffer != nullptr){aiclrtFree(dst_buffer);}
	ret = aiclrtDestroyStream(stream);				if(ret!=AICL_RET_SUCCESS){printf("destroy stream fail! \n");}									else{printf("destroy stream success \n");}
	ret = aiclrtDestroyContext(context);				if(ret!=AICL_RET_SUCCESS){printf("destroy context fail! \n");}								else{printf("destroy context success \n");}
	ret = aiclrtResetDevice(device_id);				if(ret!=AICL_RET_SUCCESS){printf("reset device fail! \n");}									else{printf("reset device success \n");}
	ret = aiclFinalize();							if(ret!=AICL_RET_SUCCESS){printf("aclFinalize fail! \n");}					    			else{printf("aclFinalize success \n");}
	return AICL_RET_SUCCESS;
}

aiclRet copy_to_device()
{
	ret = aiclrtMemcpyAsync(dst_buffer, copy_len, src_buffer, copy_len, AICL_MEMCPY_HOST_TO_DEVICE, stream);
	if(ret != AICL_RET_SUCCESS){printf("Memcpy Async  fail! \n");return -1;} 																		else{printf("destroy stream success \n");};
	
	ret = aiclrtSynchronizeStream(stream);			if(ret!=AICL_RET_SUCCESS){printf("sync stream fail! %d \n",ret);return -1;}					else{printf("sync stream success \n");};
	return AICL_RET_SUCCESS;
}

// l1 用例设计尽量简单,单个文件 解释清楚, 如果使用公共函数需要在头文件出写注释.
// 根本用例描述如何使用异步拷贝功能将数据拷贝到到device上
aiclRet main(int argc,char *argv[]){
	// 准备工作 封装到函数中 明确准备内容
	if(init_fromwork() != AICL_RET_SUCCESS){printf("main init_fromwork failed !\n");return AICL_RET_NOT_INITIALIZED;}								else{printf("main init_fromwork success \n");}

	// 要展示的功能可以封装函数也可以直接放到main函数中直接写突出展示内容和返回效果.
	//展示拷贝功能
	if(copy_to_device() != AICL_RET_SUCCESS){printf("copy_to_device failed !\n");return AICL_RET_NOT_INITIALIZED;}									else{printf("memcopy to device success \n");}

	// 收尾工作 封装到函数中 明确资源回收和其他动作
	if(uninit_formwork() !=AICL_RET_SUCCESS){printf("uninit_formwork fail! \n");return -1;}														else{printf("uninit_formwork success\n");}
	return AICL_RET_SUCCESS;
}
