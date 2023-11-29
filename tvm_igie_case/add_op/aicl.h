/**
* @file aicl.h
*
* Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef INC_EXTERNAL_AICL_H_
#define INC_EXTERNAL_AICL_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define AICL_FUNC_VISIBILITY _declspec(dllexport)
#else
#define AICL_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define AICL_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define AICL_FUNC_VISIBILITY
#endif
#endif

typedef void *aiclrtStream;
typedef void *aiclrtEvent;
typedef void *aiclrtContext;
typedef int aiclRet;

typedef uint16_t aiclFloat16;
typedef struct aiclDataBuffer aiclDataBuffer;
typedef struct aiclTensorDesc aiclTensorDesc;
typedef struct aiclmdlDataset aiclmdlDataset;
typedef struct aiclmdlDesc aiclmdlDesc;
typedef struct aiclmdlConfigHandle aiclmdlConfigHandle;
typedef struct aiclopHandle aiclopHandle;
typedef struct aiclopAttr aiclopAttr;
typedef struct aiclmdlIODims aiclmdlIODims;
typedef void (*aiclrtCallback)(void *userData);

#define AICL_MAX_DIM_CNT          128
#define AICL_MAX_TENSOR_NAME_LEN  128

typedef enum aiclrtRunMode {
    AICL_DEVICE,
    AICL_HOST
} aiclrtRunMode;

typedef enum aiclrtEventRecordedStatus {
    AICL_EVENT_RECORDED_STATUS_NOT_READY = 0,
    AICL_EVENT_RECORDED_STATUS_COMPLETE = 1
} aiclrtEventRecordedStatus;

typedef enum aiclrtEventWaitStatus {
    AICL_EVENT_WAIT_STATUS_COMPLETE  = 0,
    AICL_EVENT_WAIT_STATUS_NOT_READY = 1,
    AICL_EVENT_WAIT_STATUS_RESERVED  = 0xffff
} aiclrtEventWaitStatus;

typedef enum aiclrtCallbackBlockType {
    AICL_CALLBACK_NO_BLOCK,
    AICL_CALLBACK_BLOCK
} aiclrtCallbackBlockType;

typedef enum aiclrtMemcpyKind {
    AICL_MEMCPY_HOST_TO_HOST,
    AICL_MEMCPY_HOST_TO_DEVICE,
    AICL_MEMCPY_DEVICE_TO_HOST,
    AICL_MEMCPY_DEVICE_TO_DEVICE
} aiclrtMemcpyKind;

typedef enum aiclrtMemMallocPolicy {
    AICL_MEM_MALLOC_HUGE_FIRST,
    AICL_MEM_MALLOC_HUGE_ONLY,
    AICL_MEM_MALLOC_NORMAL_ONLY,
    AICL_MEM_MALLOC_HUGE_FIRST_P2P,
    AICL_MEM_MALLOC_HUGE_ONLY_P2P,
    AICL_MEM_MALLOC_NORMAL_ONLY_P2P
} aiclrtMemMallocPolicy;

typedef enum aiclrtMemAttr {
    AICL_DDR_MEM,
    AICL_HBM_MEM,
    AICL_DDR_MEM_HUGE,
    AICL_DDR_MEM_NORMAL,
    AICL_HBM_MEM_HUGE,
    AICL_HBM_MEM_NORMAL,
    AICL_DDR_MEM_P2P_HUGE,
    AICL_DDR_MEM_P2P_NORMAL,
    AICL_HBM_MEM_P2P_HUGE,
    AICL_HBM_MEM_P2P_NORMAL
} aiclrtMemAttr;

typedef enum aiclEngineType {
    AICL_ENGINE_SYS
} aiclopEngineType;

#define AICL_TENSOR_SHAPE_RANGE_NUM 2
#define AICL_TENSOR_VALUE_RANGE_NUM 2
#define AICL_UNKNOWN_RANK 0xFFFFFFFFFFFFFFFE

typedef enum {
    AICL_DT_UNDEFINED = -1,
    AICL_FLOAT = 0,
    AICL_FLOAT16 = 1,
    AICL_INT8 = 2,
    AICL_INT32 = 3,
    AICL_UINT8 = 4,
    AICL_INT16 = 6,
    AICL_UINT16 = 7,
    AICL_UINT32 = 8,
    AICL_INT64 = 9,
    AICL_UINT64 = 10,
    AICL_DOUBLE = 11,
    AICL_BOOL = 12,
    AICL_STRING = 13,
    AICL_COMPLEX64 = 16,
    AICL_COMPLEX128 = 17,
    AICL_BF16 = 27
} aiclDataType;

typedef enum {
    AICL_FORMAT_UNDEFINED = -1,
    AICL_FORMAT_NCHW = 0,
    AICL_FORMAT_NHWC = 1,
    AICL_FORMAT_ND = 2
} aiclFormat;

typedef enum {
    AICL_MEMTYPE_DEVICE = 0,
    AICL_MEMTYPE_HOST = 1
} aiclMemType;

typedef enum {
    AICL_MDL_PRIORITY_INT32 = 0,
    AICL_MDL_LOAD_TYPE_SIZET,
    AICL_MDL_PATH_PTR, /**< pointer to model load path with deep copy */
    AICL_MDL_MEM_ADDR_PTR, /**< pointer to model memory with shallow copy */
    AICL_MDL_MEM_SIZET,
    AICL_MDL_WEIGHT_ADDR_PTR, /**< pointer to weight memory of model with shallow copy */
    AICL_MDL_WEIGHT_SIZET,
    AICL_MDL_WORKSPACE_ADDR_PTR, /**< pointer to worksapce memory of model with shallow copy */
    AICL_MDL_WORKSPACE_SIZET
} aiclmdlConfigAttr;

typedef enum aiclCompileType {
    AICL_COMPILE_SYS,
    AICL_COMPILE_UNREGISTERED
} aiclopCompileType;

typedef enum aiclCompileFlag {
    AICL_OP_COMPILE_DEFAULT
} aiclOpCompileFlag;

static const int AICL_RET_SUCCESS = 0;
static const int AICL_RET_ERROR = -1;
static const int AICL_RET_INVALID_DEVICE = -2;
static const int AICL_RET_INVALID_DEVICE_ID = -3;
static const int AICL_RET_REPEAT_INITIALIZE = -4;
static const int AICL_RET_INVALID_FILE = -5;
static const int AICL_RET_WRITE_FILE_FAILURE = -6;
static const int AICL_RET_INVALID_FILE_SIZE = -7;
static const int AICL_RET_PARSE_FILE_FAILURE = -8;
static const int AICL_RET_INVALID_FILE_ATTR = -9;
static const int AICL_RET_INVALID_MODEL_ID = -10;
static const int AICL_RET_DESERIALIZE_MODEL_FAILURE = -11;
static const int AICL_RET_PARSE_MODEL_FAILURE = -12;
static const int AICL_RET_READ_MODEL_FAILURE = -13;
static const int AICL_RET_INVALID_MODEL_SIZE = -14;
static const int AICL_RET_INVALID_MODEL_ATTR = -15;
static const int AICL_RET_INVALID_MODEL_INPUT = -16;
static const int AICL_RET_INVALID_MODEL_OUTPUT = -17;
static const int AICL_RET_INVALID_DYNAMIC_MODEL = -18;
static const int AICL_RET_INVALID_OP_TYPE = -19;
static const int AICL_RET_INVALID_OP_INPUT = -20;
static const int AICL_RET_INVALID_OP_OUTPUT = -21;
static const int AICL_RET_INVALID_OP_ATTR = -22;
static const int AICL_RET_OP_NOT_FOUND = -23;
static const int AICL_RET_OP_LOAD_FAILED = -24;
static const int AICL_RET_INVALID_DATA_TYPE = -25;
static const int AICL_RET_INVALID_FORMAT = -26;
static const int AICL_RET_OP_COMPILE_FALURE = -27;
static const int AICL_RET_INVALID_QUEUE_ID = -28;
static const int AICL_RET_INVALID_REPEAT_SUBSCRIBE = -29;
static const int AICL_RET_INVALID_REPEAT_FINALIZE = -30;
static const int AICL_RET_INVALID_COMPILING_LIB = -31;
static const int AICL_RET_INVALID_OP_QUEUE_CONFIG = -32;
static const int AICL_RET_INVALID_OP_PATH = -33;
static const int AICL_RET_OP_STATIC_ONLY = -34;
static const int AICL_RET_RESOURCE_NOT_RELEASED = -35;
static const int AICL_RET_MEM_ALLOC_FAILURE = -36;
static const int AICL_RET_INVALID_MEM_TYPE = -37;
static const int AICL_RET_MEMORY_ADDR_UNALIGNED = -38;
static const int AICL_RET_RESOURCE_NOT_MATCH = -39;
static const int AICL_RET_INVALID_RESOURCE_HANDLE = -40;
static const int AICL_RET_FEATURE_UNSUPPORTED = -41;
static const int AICL_RET_NOT_INITIALIZED = -42;
static const int AICL_RET_INVALID_CONTEXT = -43;
static const int AICL_RET_INVALID_STREAM = -44;
static const int AICL_RET_INVALID_MODEL = -45;
static const int AICL_RET_INVALID_EVENT_TIMESTAMP = -46;
static const int AICL_RET_READ_FILE_FAILURE = -47;
static const int AICL_RET_INVALID_STREAM_SUBSCRIBE = -48;
static const int AICL_RET_INVALID_THREAD_SUBSCRIBE = -49;
static const int AICL_RET_CALLBACK_STREAM_UNREGISTERED = -50;
static const int AICL_RET_INVALID_API = -51;
static const int AICL_RET_INVALID_HANDLE = -52;
static const int AICL_RET_API_TIMEOUT = -53;
static const int AICL_RET_MEMORY_FREE_FAILURE = -54;
static const int AICL_RET_OP_OVERFLOW = -55;
static const int AICL_RET_INVALID_PARAM = -56;
static const int AICL_RET_PERMISSION_DENIED = -57;
static const int AICL_RET_NO_EVENT_RESOURCE = -58;
static const int AICL_RET_NO_STREAM_RESOURCE = -59;
static const int AICL_RET_NO_NOTIFY_RESOURCE = -60;
static const int AICL_RET_NO_MODEL_RESOURCE = -61;

/**
 * @ingroup AICL
 * @brief aicl initialize
 *
 * @par Restriction
 * The aiclInit interface can be called only once in a process
 * @param configPath [IN]    the config path,it can be NULL
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclInit(const char *configPath);

/**
 * @ingroup AICL
 * @brief aicl finalize
 *
 * @par Restriction
 * Need to call aiclFinalize before the process exits.
 * After calling aiclFinalize,the services cannot continue to be used normally.
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclFinalize();

/**
 * @ingroup AICL
 * @brief query AICL interface version
 *
 * @param majorVersion[OUT] AICL interface major version
 * @param minorVersion[OUT] AICL interface minor version
 * @param patchVersion[OUT] AICL interface patch version
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtGetVersion(int32_t *majorVersion, int32_t *minorVersion, int32_t *patchVersion);

/**
 * @ingroup AICL
 * @brief get soc name
 *
 * @retval null for failed
 * @retval OtherValues success
*/
AICL_FUNC_VISIBILITY const char *aiclrtGetSocName();

/**
 * @ingroup AICL
 * @brief Specify the device to use for the operation
 * implicitly create the default context and the default stream
 *
 * @par Function
 * The following use cases are supported:
 * @li Device can be specified in the process or thread.
 * If you call the aiclrtSetDevice interface multiple
 * times to specify the same device,
 * you only need to call the aiclrtResetDevice interface to reset the device.
 * @li The same device can be specified for operation
 *  in different processes or threads.
 * @li Device is specified in a process,
 * and multiple threads in the process can share this device to explicitly
 * create a Context (aiclrtCreateContext interface).
 * @li In multi-device scenarios, you can switch to other devices
 * through the aiclrtSetDevice interface in the process.
 *
 * @param  deviceId [IN]  the device id
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtResetDevice |aiclrtCreateContext
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSetDevice(int32_t deviceId);

/**
 * @ingroup AICL
 * @brief Reset the current operating Device and free resources on the device,
 * including the default context, the default stream,
 * and all streams created under the default context,
 * and synchronizes the interface.
 * If the task under the default context or stream has not been completed,
 * the system will wait for the task to complete before releasing it.
 *
 * @par Restriction
 * @li The Context, Stream, and Event that are explicitly created
 * on the device to be reset. Before resetting,
 * it is recommended to follow the following interface calling sequence,
 * otherwise business abnormalities may be caused.
 * @li Interface calling sequence:
 * call aiclrtDestroyEvent interface to release Event or
 * call aiclrtDestroyStream interface to release explicitly created Stream->
 * call aiclrtDestroyContext to release explicitly created Context->
 * call aiclrtResetDevice interface
 *
 * @param  deviceId [IN]   the device id
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtResetDevice(int32_t deviceId);

/**
 * @ingroup AICL
 * @brief get target device of current thread
 *
 * @param deviceId [OUT]  the device id
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtGetDevice(int32_t *deviceId);

/**
 * @ingroup AICL
 * @brief get target side
 *
 * @param runMode [OUT]    the run mode
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtGetRunMode(aiclrtRunMode *runMode);

/**
 * @ingroup AICL
 * @brief create context and associates it with the calling thread
 *
 * @par Function
 * The following use cases are supported:
 * @li If you don't call the aiclrtCreateContext interface
 * to explicitly create the context,
 * the system will use the default context, which is implicitly created
 * when the aiclrtSetDevice interface is called.
 * @li If multiple contexts are created in a process
 * (there is no limit on the number of contexts),
 * the current thread can only use one of them at the same time.
 * It is recommended to explicitly specify the context of the current thread
 * through the aiclrtSetCurrentContext interface to increase.
 * the maintainability of the program.
 *
 * @param  context [OUT]    point to the created context
 * @param  deviceId [IN]    device to create context on
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtSetDevice | aiclrtSetCurrentContext
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtCreateContext(aiclrtContext *context, int32_t deviceId);

/**
 * @ingroup AICL
 * @brief destroy context instance
 *
 * @par Function
 * Can only destroy context created through aiclrtCreateContext interface
 *
 * @param  context [IN]   the context to destroy
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtCreateContext
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtDestroyContext(aiclrtContext context);

/**
 * @ingroup AICL
 * @brief set the context of the thread
 *
 * @par Function
 * The following scenarios are supported:
 * @li If the aiclrtCreateContext interface is called in a thread to explicitly
 * create a Context (for example: ctx1), the thread's Context can be specified
 * without calling the aiclrtSetCurrentContext interface.
 * The system uses ctx1 as the context of thread1 by default.
 * @li If the aiclrtCreateContext interface is not explicitly created,
 * the system uses the default context as the context of the thread.
 * At this time, the aiclrtDestroyContext interface cannot be used to release
 * the default context.
 * @li If the aiclrtSetCurrentContext interface is called multiple times to
 * set the thread's Context, the last one prevails.
 *
 * @par Restriction
 * @li If the cevice corresponding to the context set for the thread
 * has been reset, you cannot set the context as the context of the thread,
 * otherwise a business exception will result.
 * @li It is recommended to use the context created in a thread.
 * If the aiclrtCreateContext interface is called in thread A to create a context,
 * and the context is used in thread B,
 * the user must guarantee the execution order of tasks in the same stream
 * under the same context in two threads.
 *
 * @param  context [IN]   the current context of the thread
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtCreateContext | aiclrtDestroyContext
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSetCurrentContext(aiclrtContext context);

/**
 * @ingroup AICL
 * @brief get the context of the thread
 *
 * @par Function
 * If the user calls the aiclrtSetCurrentContext interface
 * multiple times to set the context of the current thread,
 * then the last set context is obtained
 *
 * @param  context [OUT]   the current context of the thread
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtSetCurrentContext
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtGetCurrentContext(aiclrtContext *context);

/**
 * @ingroup AICL
 * @brief  create stream instance
 *
 * @param  stream [OUT]   the created stream
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtCreateStream(aiclrtStream *stream);

/**
 * @ingroup AICL
 * @brief destroy stream instance
 *
 * @par Function
 * Can only destroy streams created through the aiclrtCreateStream interface
 *
 * @par Restriction
 * Before calling the aiclrtDestroyStream interface to destroy
 * the specified Stream, you need to call the aiclrtSynchronizeStream interface
 * to ensure that the tasks in the Stream have been completed.
 *
 * @param stream [IN]  the stream to destroy
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtCreateStream | aiclrtSynchronizeStream
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtDestroyStream(aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief get total device number.
 *
 * @param count [OUT]    the device number
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtGetDeviceCount(uint32_t *count);

/**
 * @ingroup AICL
 * @brief create event instance
 *
 * @param event [OUT]   created event
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtCreateEvent(aiclrtEvent *event);

/**
 * @ingroup AICL
 * @brief create event instance with flag
 *
 * @param event [OUT]   created event
 * @param flag [IN]     event flag
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtCreateEventWithFlag(aiclrtEvent *event, uint32_t flag);

/**
 * @ingroup AICL
 * @brief destroy event instance
 *
 * @par Function
 *  Only events created through the aiclrtCreateEvent interface can be
 *  destroyed, synchronous interfaces. When destroying an event,
 *  the user must ensure that the tasks involved in the aiclrtSynchronizeEvent
 *  interface or the aiclrtStreamWaitEvent interface are completed before
 *  they are destroyed.
 *
 * @param  event [IN]   event to destroy
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtCreateEvent | aiclrtSynchronizeEvent | aiclrtStreamWaitEvent
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtDestroyEvent(aiclrtEvent event);

/**
 * @ingroup AICL
 * @brief Record an Event in the Stream
 *
 * @param event [IN]    event to record
 * @param stream [IN]   stream handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtRecordEvent(aiclrtEvent event, aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief Reset an event
 *
 * @par Function
 *  Users need to make sure to wait for the tasks in the Stream
 *  to complete before resetting the Event
 *
 * @param event [IN]    event to reset
 * @param stream [IN]   stream handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtResetEvent(aiclrtEvent event, aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief Queries an event's status
 *
 * @param  event [IN]    event to query
 * @param  status [OUT]  event recorded status
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtQueryEventStatus(aiclrtEvent event, aiclrtEventRecordedStatus *status);

/**
* @ingroup AICL
* @brief Queries an event's wait-status
*
* @param  event [IN]    event to query
* @param  status [OUT]  event wait-status
*
* @retval AICL_RET_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*/
AICL_FUNC_VISIBILITY aiclRet aiclrtQueryEventWaitStatus(aiclrtEvent event, aiclrtEventWaitStatus *status);

/**
 * @ingroup AICL
 * @brief Block Host Running, wait event to be complete
 *
 * @param  event [IN]   event to wait
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSynchronizeEvent(aiclrtEvent event);

/**
 * @ingroup AICL
 * @brief computes the elapsed time between events.
 *
 * @param ms [OUT]     time between start and end in ms
 * @param start [IN]   starting event
 * @param end [IN]     ending event
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtCreateEvent | aiclrtRecordEvent | aiclrtSynchronizeStream
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtEventElapsedTime(float *ms, aiclrtEvent startEvent, aiclrtEvent endEvent);

/**
 * @ingroup AICL
 * @brief Blocks the operation of the specified Stream until
 * the specified Event is completed.
 * Support for multiple streams waiting for the same event.
 *
 * @param  stream [IN]   the wait stream If using thedefault Stream, set NULL
 * @param  event [IN]    the event to wait
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtStreamWaitEvent(aiclrtStream stream, aiclrtEvent event);

/**
 * @ingroup AICL
 * @brief Wait for compute device to finish
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSynchronizeDevice(void);

/**
 * @ingroup AICL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSynchronizeStream(aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief The thread that handles the callback function on the Stream
 *
 * @param threadId [IN] thread ID
 * @param stream [IN]   stream handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSubscribeReport(uint64_t threadId, aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief Add a callback function to be executed on the host
 *        to the task queue of the Stream
 *
 * @param fn [IN]   Specify the callback function to be added
 *                  The function prototype of the callback function is:
 *                  typedef void (*aiclrtCallback)(void *userData);
 * @param userData [IN]   User data to be passed to the callback function
 * @param blockType [IN]  callback block type
 * @param stream [IN]     stream handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtLaunchCallback(aiclrtCallback fn, void *userData, aiclrtCallbackBlockType blockType,
                                                 aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief alloc memory on device
 *
 * @par Function
 *  alloc for size linear memory on device
 *  and return a pointer to allocated memory by *devPtr
 *
 * @par Restriction
 * @li The memory requested by the aiclrtMalloc interface needs to be released
 * through the aiclrtFree interface.
 * @li Before calling the media data processing interface,
 * if you need to apply memory on the device to store input or output data,
 * you need to call aicldvppMalloc to apply for memory.
 *
 * @param devPtr [OUT]  pointer to pointer to allocated memory on device
 * @param size [IN]     alloc memory size
 * @param policy [IN]   memory alloc policy
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtFree | aicldvppMalloc | aiclrtMallocCached
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtMalloc(void **devPtr,
                                         size_t size,
                                         aiclrtMemMallocPolicy policy);

/**
 * @ingroup AICL
 * @brief free device memory
 *
 * @par Function
 *  can only free memory allocated through the aiclrtMalloc interface
 *
 * @param  devPtr [IN]  Pointer to memory to be freed
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtMalloc
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtFree(void *devPtr);

/**
 * @ingroup AICL
 * @brief alloc memory on host
 *
 * @par Restriction
 * @li The requested memory cannot be used in the Device
 * and needs to be explicitly copied to the Device.
 * @li The memory requested by the aiclrtMallocHost interface
 * needs to be released through the aiclrtFreeHost interface.
 *
 * @param  hostPtr [OUT] pointer to pointer to allocated memory on the host
 * @param  size [IN]     alloc memory size
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtFreeHost
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtMallocHost(void **hostPtr, size_t size);

/**
 * @ingroup AICL
 * @brief free host memory
 *
 * @par Function
 *  can only free memory allocated through the aiclrtMallocHost interface
 *
 * @param  hostPtr [IN]   free memory pointer
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtMallocHost
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtFreeHost(void *hostPtr);

/**
 * @ingroup AICL
 * @brief synchronous memory replication between host and device
 *
 * @param dst [IN]       destination address pointer
 * @param destMax [IN]   Max length of the destination address memory
 * @param src [IN]       source address pointer
 * @param count [IN]     the length of byte to copy
 * @param kind [IN]      memcpy type
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtMemcpy(void *dst,
                                         size_t destMax,
                                         const void *src,
                                         size_t count,
                                         aiclrtMemcpyKind kind);

/**
 * @ingroup AICL
 * @brief Initialize memory and set contents of memory to specified value
 *
 * @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
 * @param devPtr [IN]    Starting address of memory
 * @param maxCount [IN]  Max length of destination address memory
 * @param value [IN]     Set value
 * @param count [IN]     The length of memory
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count);

/**
 * @ingroup AICL
 * @brief  Asynchronous memory replication between Host and Device
 *
 * @par Function
 *  After calling this interface,
 *  be sure to call the aiclrtSynchronizeStream interface to ensure that
 *  the task of memory replication has been completed
 *
 * @par Restriction
 * @li For on-chip Device-to-Device memory copy,
 *     both the source and destination addresses must be 64-byte aligned
 *
 * @param dst [IN]     destination address pointer
 * @param destMax [IN] Max length of destination address memory
 * @param src [IN]     source address pointer
 * @param count [IN]   the number of byte to copy
 * @param kind [IN]    memcpy type
 * @param stream [IN]  asynchronized task stream
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtSynchronizeStream
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtMemcpyAsync(void *dst,
                                              size_t destMax,
                                              const void *src,
                                              size_t count,
                                              aiclrtMemcpyKind kind,
                                              aiclrtStream stream);
/**
* @ingroup AICL
* @brief Asynchronous initialize memory
* and set contents of memory to specified value async
*
* @par Function
 *  The memory to be initialized is on the Host or device side,
 *  and the system determines whether
 *  it is host or device according to the address
 *
* @param devPtr [IN]      destination address pointer
* @param maxCount [IN]    Max length of destination address memory
* @param value [IN]       set value
* @param count [IN]       the number of byte to set
* @param stream [IN]      asynchronized task stream
*
* @retval AICL_RET_SUCCESS The function is successfully executed.
* @retval OtherValues Failure
*
* @see aiclrtSynchronizeStream
*/
AICL_FUNC_VISIBILITY aiclRet aiclrtMemsetAsync(void *devPtr,
                                              size_t maxCount,
                                              int32_t value,
                                              size_t count,
                                              aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief Obtain the free memory and total memory of specified attribute.
 * the specified memory include normal memory and huge memory.
 *
 * @param attr [IN]    the memory attribute of specified device
 * @param free [OUT]   the free memory of specified device
 * @param total [OUT]  the total memory of specified device.
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtGetMemInfo(aiclrtMemAttr attr, size_t *free, size_t *total);

/**
 * @ingroup AICL
 * @brief Get the number of aiclDataBuffer in aiclmdlDataset
 *
 * @param dataset [IN]   aiclmdlDataset pointer
 *
 * @retval the number of aiclDataBuffer
 */
AICL_FUNC_VISIBILITY size_t aiclmdlGetDatasetNumBuffers(const aiclmdlDataset *dataset);

/**
 * @ingroup AICL
 * @brief Get the aiclDataBuffer in aiclmdlDataset by index
 *
 * @param dataset [IN]   aiclmdlDataset pointer
 * @param index [IN]     the index of aiclDataBuffer
 *
 * @retval Get successfully, return the address of aiclDataBuffer
 * @retval Failure return NULL
 */
AICL_FUNC_VISIBILITY aiclDataBuffer *aiclmdlGetDatasetBuffer(const aiclmdlDataset *dataset, size_t index);

/**
 * @ingroup AICL
 * @brief Load offline model data from files
 * and manage memory internally by the system
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 *
 * @param modelPath [IN]   Storage path for offline model files
 * @param modelId [OUT]    Model ID generated after
 *        the system finishes loading the model
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlLoadFromFile(const char *modelPath, uint32_t *modelId);

/**
 * @ingroup AICL
 * @brief Load offline model data from memory and manage the memory of
 * model running internally by the system
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 *
 * @param model [IN]      Model data stored in memory
 * @param modelSize [IN]  model data size
 * @param modelId [OUT]   Model ID generated after
 *        the system finishes loading the model
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlLoadFromMem(const void *model,  size_t modelSize,
                                               uint32_t *modelId);

/**
 * @ingroup AICL
 * @brief Load offline model data from a file,
 * and the user manages the memory of the model run by itself
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations.
 * @param modelPath [IN]   Storage path for offline model files
 * @param modelId [OUT]    Model ID generated after finishes loading the model
 * @param workPtr [IN]     A pointer to the working memory
 *                         required by the model on the Device,can be null
 * @param workSize [IN]    The amount of working memory required by the model
 * @param weightPtr [IN]   Pointer to model weight memory on Device
 * @param weightSize [IN]  The amount of weight memory required by the model
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlLoadFromFileWithMem(const char *modelPath,
                                                       uint32_t *modelId, void *workPtr, size_t workSize,
                                                       void *weightPtr, size_t weightSize);

/**
 * @ingroup AICL
 * @brief Load offline model data from memory,
 * and the user can manage the memory of model running
 *
 * @par Function
 * After the system finishes loading the model,
 * the model ID returned is used as a mark to identify the model
 * during subsequent operations
 * @param model [IN]      Model data stored in memory
 * @param modelSize [IN]  model data size
 * @param modelId [OUT]   Model ID generated after finishes loading the model
 * @param workPtr [IN]    A pointer to the working memory
 *                        required by the model on the Device,can be null
 * @param workSize [IN]   work memory size
 * @param weightPtr [IN]  Pointer to model weight memory on Device,can be null
 * @param weightSize [IN] The amount of weight memory required by the model
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlLoadFromMemWithMem(const void *model, size_t modelSize,
                                                      uint32_t *modelId, void *workPtr, size_t workSize,
                                                      void *weightPtr, size_t weightSize);

/**
 * @ingroup AICL
 * @brief Execute model synchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlExecute(uint32_t modelId, const aiclmdlDataset *input, aiclmdlDataset *output);

/**
 * @ingroup AICL
 * @brief Execute model asynchronous inference until the inference result is returned
 *
 * @param  modelId [IN]   ID of the model to perform inference
 * @param  input [IN]     Input data for model inference
 * @param  output [OUT]   Output data for model inference
 * @param  stream [IN]    stream
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclmdlLoadFromFile | aiclmdlLoadFromMem | aiclmdlLoadFromFileWithMem |
 * aiclmdlLoadFromMemWithMem
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlExecuteAsync(uint32_t modelId, const aiclmdlDataset *input,
                                                aiclmdlDataset *output, aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief unload model with model id
 *
 * @param  modelId [IN]   model id to be unloaded
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlUnload(uint32_t modelId);

/**
 * @ingroup AICL
 * @brief Get the weight memory size and working memory size
 * required for model execution according to the model file
 *
 * @param  fileName [IN]     Model path to get memory information
 * @param  workSize [OUT]    The amount of working memory for model executed
 * @param  weightSize [OUT]  The amount of weight memory for model executed
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize);

/**
 * @ingroup AICL
 * @brief Obtain the weights required for
 * model execution according to the model data in memory
 *
 * @par Restriction
 * The execution and weight memory is Device memory,
 * and requires user application and release.
 * @param  model [IN]        model memory which user manages
 * @param  modelSize [IN]    model data size
 * @param  workSize [OUT]    The amount of working memory for model executed
 * @param  weightSize [OUT]  The amount of weight memory for model executed
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlQuerySizeFromMem(const void *model, size_t modelSize, size_t *workSize,
                                                    size_t *weightSize);

/**
 * @ingroup AICL
 * @brief create model config handle of type aiclmdlConfigHandle
 *
 * @retval the aiclmdlConfigHandle pointer
 *
 * @see aiclmdlDestroyConfigHandle
*/
AICL_FUNC_VISIBILITY aiclmdlConfigHandle *aiclmdlCreateConfigHandle();

/**
 * @ingroup AICL
 * @brief destroy data of type aiclmdlConfigHandle
 *
 * @param handle [IN]   pointer to model config handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclmdlCreateConfigHandle
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlDestroyConfigHandle(aiclmdlConfigHandle *handle);

/**
 * @ingroup AICL
 * @brief set config for model load
 *
 * @param handle [OUT]    pointer to model config handle
 * @param attr [IN]       config attr in model config handle to be set
 * @param attrValue [IN]  pointer to model config value
 * @param valueSize [IN]  memory size of attrValue
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlSetConfigOpt(aiclmdlConfigHandle *handle, aiclmdlConfigAttr attr,
    const void *attrValue, size_t valueSize);

/**
 * @ingroup AICL
 * @brief load model with config
 *
 * @param handle [IN]    pointer to model config handle
 * @param modelId [OUT]  pointer to model id
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
*/
AICL_FUNC_VISIBILITY aiclRet aiclmdlLoadWithConfig(const aiclmdlConfigHandle *handle, uint32_t *modelId);

/**
 * @ingroup AICL
 * @brief compile op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param attr [IN]           pointer to instance of aiclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopCompile(const char *opType,
                                          int numInputs,
                                          const aiclTensorDesc *const inputDesc[],
                                          int numOutputs,
                                          const aiclTensorDesc *const outputDesc[],
                                          const aiclopAttr *attr,
                                          aiclopEngineType engineType,
                                          aiclopCompileType compileFlag,
                                          const char *opPath);

/**
 * @ingroup AICL
 * @brief set compile flag
 *
 * @param flag [IN]    compile flag, AICL_OP_COMPILE_DEFAULT means compile with default mode
 *                     AICL_OP_COMPILE_FUZZ means compile with fuzz mode
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetCompileFlag(aiclOpCompileFlag flag);

/**
 * @ingroup AICL
 * @brief Set base directory that contains single op models
 *
 * @par Restriction
 * The aiclopSetModelDir interface can be called only once in a process.
 * @param  modelDir [IN]   path of the directory
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetModelDir(const char *modelDir);

/**
 * @ingroup AICL
 * @brief load single op models from memory
 *
 * @par Restriction
 * The aiclopLoad interface can be called more than one times in a process.
 * @param model [IN]        address of single op models
 * @param modelSize [IN]    size of single op models
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopLoad(const void *model, size_t modelSize);

/**
 * @ingroup AICL
 * @brief compile and execute op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN|OUT]   pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aiclopAttr.
 *                              may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 * @param stream [IN]           stream handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopCompileAndExecute(const char *opType,
    int numInputs, aiclTensorDesc *inputDesc[], aiclDataBuffer *inputs[],
    int numOutputs, aiclTensorDesc *outputDesc[], aiclDataBuffer *outputs[],
    aiclopAttr *attr, aiclopEngineType engineType, aiclopCompileType compileFlag,
    const char *opPath, aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief create a instance of aiclopHandle.
 *
 * @param opType [IN]      type of op
 * @param numInputs [IN]   number of inputs
 * @param inputDesc [IN]   pointer to array of input tensor descriptions
 * @param numOutputs [IN]  number of outputs
 * @param outputDesc [IN]  pointer to array of output tensor descriptions
 * @param opAttr [IN]      pointer to instance of aiclopAttr.
 *                         may pass nullptr if the op has no attribute
 * @param handle [OUT]     pointer to the pointer to the handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopCreateHandle(const char *opType,
                                               int numInputs,
                                               const aiclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aiclTensorDesc *const outputDesc[],
                                               const aiclopAttr *opAttr,
                                               aiclopHandle **handle);

/**
 * @ingroup AICL
 * @brief destroy aiclopHandle instance
 *
 * @param handle [IN]   pointer to the instance of aiclopHandle
 */
AICL_FUNC_VISIBILITY void aiclopDestroyHandle(aiclopHandle *handle);

/**
 * @ingroup AICL
 * @brief execute an op with the handle.
 *        can save op model matching cost compared with aiclopExecute
 *
 * @param handle [IN]      pointer to the instance of aiclopHandle.
 *                         The aiclopCreateHandle interface has been called
 *                         in advance to create aiclopHandle type data.
 * @param numInputs [IN]   number of inputs
 * @param inputs [IN]      pointer to array of input buffers.
 *                         The aiclCreateDataBuffer interface has been called
 *                         in advance to create aiclDataBuffer type data.
 * @param numOutputs [IN]  number of outputs
 * @param outputs [OUT]    pointer to array of output buffers
 * @param stream [IN]      stream
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclopCreateHandle | aiclCreateDataBuffer
 */
AICL_FUNC_VISIBILITY aiclRet aiclopExecWithHandle(aiclopHandle *handle,
                                                 int numInputs,
                                                 const aiclDataBuffer *const inputs[],
                                                 int numOutputs,
                                                 aiclDataBuffer *const outputs[],
                                                 aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief inferShape the specified operator synchronously
 *
 * @param opType [IN]       type of op
 * @param numInputs [IN]    number of inputs
 * @param inputDesc [IN]    pointer to array of input tensor descriptions
 * @param inputs [IN]       pointer to array of input buffers
 * @param numOutputs [IN]   number of outputs
 * @param outputDesc [OUT]  pointer to array of output tensor descriptions
 * @param attr [IN]         pointer to instance of aiclopAttr.
 *                          may pass nullptr if the op has no attribute
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopInferShape(const char *opType,
                                             int numInputs,
                                             aiclTensorDesc *inputDesc[],
                                             aiclDataBuffer *inputs[],
                                             int numOutputs,
                                             aiclTensorDesc *outputDesc[],
                                             aiclopAttr *attr);

/**
 * @ingroup AICL
 * @brief create data aiclTensorDesc
 *
 * @param  dataType [IN]    Data types described by tensor
 * @param  numDims [IN]     the number of dimensions of the shape
 * @param  dims [IN]        the size of the specified dimension
 * @param  format [IN]      tensor format
 *
 * @retval aiclTensorDesc pointer.
 * @retval nullptr if param is invalid or run out of memory
 */
AICL_FUNC_VISIBILITY aiclTensorDesc *aiclCreateTensorDesc(aiclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aiclFormat format);

/**
 * @ingroup AICL
 * @brief destroy data aiclTensorDesc
 *
 * @param desc [IN]     pointer to the data of aiclTensorDesc to destroy
 */
AICL_FUNC_VISIBILITY void aiclDestroyTensorDesc(const aiclTensorDesc *desc);

/**
 * @ingroup AICL
 * @brief Set the format specified by the tensor description
 *
 * @param  desc [OUT]     pointer to the instance of aiclTensorDesc
 * @param  format [IN]    the storage format
 *
 * @retval AICL_RET_SUCCESS    The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclSetTensorFormat(aiclTensorDesc *desc, aiclFormat format);

/**
 * @ingroup AICL
 * @brief Set tensor memory type specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aiclTensorDesc
 * @param  memType [IN]       AICL_MEMTYPE_DEVICE means device, AICL_MEMTYPE_HOST or
 * AICL_MEMTYPE_HOST_COMPILE_INDEPENDENT means host
 *
 * @retval AICL_RET_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclSetTensorPlaceMent(aiclTensorDesc *desc, aiclMemType memType);

/**
 * @ingroup AICL
 * @brief Set the shape specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aiclTensorDesc
 * @param  numDims [IN]    the number of dimensions of the shape
 * @param  dims [IN]       the size of the specified dimension
 *
 * @retval AICL_RET_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclSetTensorShape(aiclTensorDesc *desc, int numDims, const int64_t *dims);

/**
 * @ingroup AICL
 * @brief set tensor shape range for aiclTensorDesc
 *
 * @param  desc [OUT]     pointer to the data of aiclTensorDesc
 * @param  dimsCount [IN]     the number of dimensions of the shape
 * @param  dimsRange [IN]     the range of dimensions of the shape
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclSetTensorShapeRange(aiclTensorDesc* desc,
                                                    size_t dimsCount,
                                                    int64_t dimsRange[][AICL_TENSOR_SHAPE_RANGE_NUM]);

/**
 * @ingroup AICL
 * @brief set tensor description name
 *
 * @param desc [OUT]       pointer to the instance of aiclTensorDesc
 * @param name [IN]        tensor description name
 */
AICL_FUNC_VISIBILITY void aiclSetTensorDescName(aiclTensorDesc *desc, const char *name);

/**
 * @ingroup AICL
 * @brief Set const data specified by the tensor description
 *
 * @param  desc [OUT]      pointer to the instance of aiclTensorDesc
 * @param  dataBuffer [IN]       pointer to the const databuffer
 * @param  length [IN]       the length of const databuffer
 *
 * @retval AICL_RET_SUCCESS     The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclSetTensorConst(aiclTensorDesc *desc, void *dataBuffer, size_t length);

/**
 * @ingroup AICL
 * @brief get data type specified by the tensor description
 *
 * @param desc [IN]        pointer to the instance of aiclTensorDesc
 *
 * @retval data type specified by the tensor description.
 * @retval AICL_DT_UNDEFINED if description is null
 */
AICL_FUNC_VISIBILITY aiclDataType aiclGetTensorDescType(const aiclTensorDesc *desc);

/**
 * @ingroup AICL
 * @brief get data format specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aiclTensorDesc
 *
 * @retval data format specified by the tensor description.
 * @retval AICL_FORMAT_UNDEFINED if description is null
 */
AICL_FUNC_VISIBILITY aiclFormat aiclGetTensorDescFormat(const aiclTensorDesc *desc);

/**
 * @ingroup AICL
 * @brief get tensor size specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aiclTensorDesc
 *
 * @retval data size specified by the tensor description.
 * @retval 0 if description is null
 */
AICL_FUNC_VISIBILITY size_t aiclGetTensorDescSize(const aiclTensorDesc *desc);

/**
 * @ingroup AICL
 * @brief get element count specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aiclTensorDesc
 *
 * @retval element count specified by the tensor description.
 * @retval 0 if description is null
 */
AICL_FUNC_VISIBILITY size_t aiclGetTensorDescElementCount(const aiclTensorDesc *desc);

/**
 * @ingroup AICL
 * @brief get number of dims specified by the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aiclTensorDesc
 *
 * @retval number of dims specified by the tensor description.
 * @retval 0 if description is null
 * @retval AICL_UNKNOWN_RANK if the tensor dim is -2
 */
AICL_FUNC_VISIBILITY size_t aiclGetTensorDescNumDims(const aiclTensorDesc *desc);

/**
 * @ingroup AICL
 * @brief create data of aiclDataBuffer
 *
 * @param data [IN]    pointer to data
 * @li Need to be managed by the user,
 *  call aiclrtMalloc interface to apply for memory,
 *  call aiclrtFree interface to release memory
 *
 * @param size [IN]    size of data in bytes
 *
 * @retval pointer to created instance. nullptr if run out of memory
 *
 * @see aiclrtMalloc | aiclrtFree
 */
AICL_FUNC_VISIBILITY aiclDataBuffer *aiclCreateDataBuffer(void *data, size_t size);

/**
 * @ingroup AICL
 * @brief destroy data of aiclDataBuffer
 *
 * @par Function
 *  Only the aiclDataBuffer type data is destroyed here.
 *  The memory of the data passed in when the aiclDataDataBuffer interface
 *  is called to create aiclDataBuffer type data must be released by the user
 *
 * @param  dataBuffer [IN]   pointer to the aiclDataBuffer
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclCreateDataBuffer
 */
AICL_FUNC_VISIBILITY aiclRet aiclDestroyDataBuffer(const aiclDataBuffer *dataBuffer);

/**
 * @ingroup AICL
 * @brief create data of type aiclopAttr
 *
 * @retval pointer to created instance.
 * @retval nullptr if run out of memory
 */
AICL_FUNC_VISIBILITY aiclopAttr *aiclopCreateAttr();

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is bool
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *                         false if attrValue is 0, true otherwise.
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrBool(aiclopAttr *attr, const char *attrName, uint8_t attrValue);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is int64_t
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrInt(aiclopAttr *attr, const char *attrName, int64_t attrValue);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is float
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrFloat(aiclopAttr *attr, const char *attrName, float attrValue);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is string
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrString(aiclopAttr *attr, const char *attrName, const char *attrValue);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is aiclDataType
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param attrValue [IN]   attribute value
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrDataType(aiclopAttr *attr, const char *attrName, aiclDataType attrValue);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is list of aiclDataType
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values. false if attrValue is 0, true otherwise.
 * @param values [IN]      pointer to values
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrListDataType(aiclopAttr *attr, const char *attrName, int numValues,
    const aiclDataType values[]);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is list of bools
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values. false if attrValue is 0, true otherwise.
 * @param values [IN]      pointer to values
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrListBool(aiclopAttr *attr, const char *attrName, int numValues,
    const uint8_t *values);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is list of ints
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrListInt(aiclopAttr *attr, const char *attrName, int numValues,
    const int64_t *values);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is list of floats
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrListFloat(aiclopAttr *attr, const char *attrName, int numValues,
    const float *values);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is list of strings
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param numValues [IN]   number of values
 * @param values [IN]      pointer to values
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrListString(aiclopAttr *attr, const char *attrName, int numValues,
    const char **values);

/**
 * @ingroup AICL
 * @brief set an attribute. the type of the attribute is list of list of ints
 *
 * @param attr [OUT]       pointer to the instance of aiclopAttr
 * @param attrName [IN]    attribute name
 * @param numLists [IN]    number of lists
 * @param numValues [IN]   pointer to number of values of each list
 * @param values [IN]      pointer to values
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopSetAttrListListInt(aiclopAttr *attr,
                                                     const char *attrName,
                                                     int numLists,
                                                     const int *numValues,
                                                     const int64_t *const values[]);

/**
 * @ingroup AICL
 * @brief Create data of type aiclmdlDataset
 *
 * @retval the aiclmdlDataset pointer
 */
AICL_FUNC_VISIBILITY aiclmdlDataset *aiclmdlCreateDataset();

/**
 * @ingroup AICL
 * @brief destroy data of type aiclmdlDataset
 *
 * @param  dataset [IN]  Pointer to aiclmdlDataset to be destroyed
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlDestroyDataset(const aiclmdlDataset *dataset);

/**
 * @ingroup AICL
 * @brief Load and execute the specified operator
 *
 * @par Restriction
 * @li The input and output organization of each operator is different,
 * and the application needs to organize the operator strictly
 * according to the operator input and output parameters when calling.
 * @li When the user calls aiclopExecute,
 * the AICL finds the corresponding task according to the optype,
 * the description of the input tesnsor,
 * the description of the output tesnsor, and attr, and issues the execution.
 *
 * @param opType [IN]      type of op
 * @param numInputs [IN]   number of inputs
 * @param inputDesc [IN]   pointer to array of input tensor descriptions
 * @param inputs [IN]      pointer to array of input buffers
 * @param numOutputs [IN]  number of outputs
 * @param outputDesc [IN|OUT]  pointer to array of output tensor descriptions
 * @param outputs [OUT]    pointer to array of output buffers
 * @param attr [IN]        pointer to instance of aiclopAttr.
 *                         may pass nullptr if the op has no attribute
 * @param stream [IN]      stream
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclopExecute(const char *opType,
                                            int numInputs,
                                            aiclTensorDesc *inputDesc[],
                                            aiclDataBuffer *inputs[],
                                            int numOutputs,
                                            aiclTensorDesc *outputDesc[],
                                            aiclDataBuffer *outputs[],
                                            aiclopAttr *attr,
                                            aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief Add aiclDataBuffer to aiclmdlDataset
 *
 * @param dataset [OUT]    aiclmdlDataset address of aiclDataBuffer to be added
 * @param dataBuffer [IN]  aiclDataBuffer address to be added
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlAddDatasetBuffer(aiclmdlDataset *dataset, aiclDataBuffer *dataBuffer);

/**
 * @ingroup AICL
 * @brief Create data of type aiclmdlDesc
 *
 * @retval the aiclmdlDesc pointer
 */
AICL_FUNC_VISIBILITY aiclmdlDesc *aiclmdlCreateDesc();

/**
 * @ingroup AICL
 * @brief destroy data of type aiclmdlDesc
 *
 * @param modelDesc [IN]   Pointer to almdldlDesc to be destroyed
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlDestroyDesc(aiclmdlDesc *modelDesc);

/**
 * @ingroup AICL
 * @brief Get aiclmdlDesc data of the model according to the model ID
 *
 * @param  modelDesc [OUT]   aiclmdlDesc pointer
 * @param  modelId [IN]      model id
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlGetDesc(aiclmdlDesc *modelDesc, uint32_t modelId);

/**
 * @ingroup AICL
 * @brief Get the number of the inputs of
 *        the model according to data of aiclmdlDesc
 *
 * @param  modelDesc [IN]   aiclmdlDesc pointer
 *
 * @retval input size with aiclmdlDesc
 */
AICL_FUNC_VISIBILITY size_t aiclmdlGetNumInputs(aiclmdlDesc *modelDesc);

/**
 * @ingroup AICL
 * @brief Get the number of the output of
 *        the model according to data of aiclmdlDesc
 *
 * @param  modelDesc [IN]   aiclmdlDesc pointer
 *
 * @retval output size with aiclmdlDesc
 */
AICL_FUNC_VISIBILITY size_t aiclmdlGetNumOutputs(aiclmdlDesc *modelDesc);

/**
 * @ingroup AICL
 * @brief Get the size of the specified input according to
 *        the data of type aiclmdlDesc
 *
 * @param  modelDesc [IN]  aiclmdlDesc pointer
 * @param  index [IN] the size of the number of inputs to be obtained,
 *         the index value starts from 0
 *
 * @retval Specify the size of the input
 */
AICL_FUNC_VISIBILITY size_t aiclmdlGetInputSizeByIndex(aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief Get the size of the specified output according to
 *        the data of type aiclmdlDesc
 *
 * @param modelDesc [IN]   aiclmdlDesc pointer
 * @param index [IN]  the size of the number of outputs to be obtained,
 *        the index value starts from 0
 *
 * @retval Specify the size of the output
 */
AICL_FUNC_VISIBILITY size_t aiclmdlGetOutputSizeByIndex(aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief destroy data of typ aiclopAttr
 *
 * @param attr [IN]   pointer to the instance of aiclopAttr
 */
AICL_FUNC_VISIBILITY void aiclopDestroyAttr(const aiclopAttr *attr);

/**
 * @ingroup AICL
 * @brief Get the size of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aiclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimSize [OUT]    size of the specified dim.
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclGetTensorDescDim(const aiclTensorDesc *desc, size_t index, int64_t *dimSize);

/**
 * @ingroup AICL
 * @brief get op description info
 *
 * @param deviceId [IN]       device id
 * @param streamId [IN]       stream id
 * @param taskId [IN]         task id
 * @param opName [OUT]        pointer to op name
 * @param opNameLen [IN]      the length of op name
 * @param inputDesc [OUT]     pointer to input description
 * @param numInputs [OUT]     the number of input tensor
 * @param outputDesc [OUT]    pointer to output description
 * @param numOutputs [OUT]    the number of output tensor
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed
 * @retval OtherValues Failure
*/
AICL_FUNC_VISIBILITY aiclRet aiclmdlCreateAndGetOpDesc(uint32_t deviceId, uint32_t streamId,
    uint32_t taskId, char *opName, size_t opNameLen, aiclTensorDesc **inputDesc, size_t *numInputs,
    aiclTensorDesc **outputDesc, size_t *numOutputs);

/**
 * @ingroup AICL
 * @brief get data address from aiclDataBuffer
 *
 * @param dataBuffer [IN]    pointer to the data of aiclDataBuffer
 *
 * @retval data address
 */
AICL_FUNC_VISIBILITY void *aiclGetDataBufferAddr(const aiclDataBuffer *dataBuffer);

/**
 * 
 * @ingroup AICL
 * @brief get data size of aiclDataBuffer to replace aiclGetDataBufferSize
 *
 * @param  dataBuffer [IN]    pointer to the data of aiclDataBuffer
 *
 * @retval data size
 */
AICL_FUNC_VISIBILITY size_t aiclGetDataBufferSize(const aiclDataBuffer *dataBuffer);

/**
 * @ingroup AICL
 * @brief get op description info
 *
 * @param desc [IN]     pointer to tensor description
 * @param index [IN]    index of tensor
 *
 * @retval null for failed.
 * @retval OtherValues success.
*/
AICL_FUNC_VISIBILITY aiclTensorDesc *aiclGetTensorDescByIndex(aiclTensorDesc *desc, size_t index);

/**
 * @ingroup AICL
 * @brief Set the timeout interval for waitting of op
 *
 * @param timeout [IN]   op wait timeout
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSetOpWaitTimeout(uint32_t timeout);

/**
 * @ingroup AICL
 * @brief get input dims info(version 2), especially for static aipp
 * it is the same with aiclmdlGetInputDims while model without static aipp
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     input tensor index
 * @param dims [OUT]     dims info
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclmdlGetInputDims
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlGetInputDims(const aiclmdlDesc *modelDesc, size_t index, aiclmdlIODims *dims);

/**
 * @ingroup AICL
 * @brief get output dims info
 *
 * @param modelDesc [IN] model description
 * @param index [IN]     output tensor index
 * @param dims [OUT]     dims info
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclmdlGetOutputDims(const aiclmdlDesc *modelDesc, size_t index, aiclmdlIODims *dims);

/**
 * @ingroup AICL
 * @brief get input name by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      intput tensor index
 *
 * @retval input tensor name,the same life cycle with modelDesc
 */
AICL_FUNC_VISIBILITY const char *aiclmdlGetInputNameByIndex(const aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief get output name by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      output tensor index
 *
 * @retval output tensor name,the same life cycle with modelDesc
 */
AICL_FUNC_VISIBILITY const char *aiclmdlGetOutputNameByIndex(const aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief get input format by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      intput tensor index
 *
 * @retval input tensor format
 */
AICL_FUNC_VISIBILITY aiclFormat aiclmdlGetInputFormat(const aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief get output format by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]      output tensor index
 *
 * @retval output tensor format
 */
AICL_FUNC_VISIBILITY aiclFormat aiclmdlGetOutputFormat(const aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief get input data type by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  intput tensor index
 *
 * @retval input tensor data type
 */
AICL_FUNC_VISIBILITY aiclDataType aiclmdlGetInputDataType(const aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief get output data type by index
 *
 * @param modelDesc [IN]  model description
 * @param index [IN]  output tensor index
 *
 * @retval output tensor data type
 */
AICL_FUNC_VISIBILITY aiclDataType aiclmdlGetOutputDataType(const aiclmdlDesc *modelDesc, size_t index);

/**
 * @ingroup AICL
 * @brief Get the range of the specified dim in the tensor description
 *
 * @param  desc [IN]        pointer to the instance of aiclTensorDesc
 * @param  index [IN]       index of dims, start from 0.
 * @param  dimRangeNum [IN]     number of dimRange.
 * @param  dimRange [OUT]       range of the specified dim.
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclGetTensorDescDimRange(const aiclTensorDesc *desc,
                                                      size_t index,
                                                      size_t dimRangeNum,
                                                      int64_t *dimRange);

/**
 * @ingroup AICL
 * @brief After waiting for a specified time, trigger callback processing
 *
 * @par Function
 *  The thread processing callback specified by
 *  the aiclrtSubscribeReport interface
 *
 * @param timeout [IN]   timeout value
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aiclrtSubscribeReport
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtProcessReport(int32_t timeout);

/**
 * @ingroup AICL
 * @brief Cancel thread registration,
 *        the callback function on the specified Stream
 *        is no longer processed by the specified thread
 *
 * @param threadId [IN]   thread ID
 * @param stream [IN]     stream handle
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtUnSubscribeReport(uint64_t threadId, aiclrtStream stream);

/**
 * @ingroup AICL
 * @brief get tensor description name
 *
 * @param  desc [IN]        pointer to the instance of aiclTensorDesc
 *
 * @retval tensor description name.
 * @retval empty string if description is null
 */
AICL_FUNC_VISIBILITY const char *aiclGetTensorDescName(aiclTensorDesc *desc);

/**
 * @ingroup AICL
 * @brief block the host until all tasks
 * in the specified stream have completed
 *
 * @param  stream [IN]   the stream to wait
 * @param  timeout [IN]  timeout value,the unit is milliseconds
 * -1 means waiting indefinitely, 0 means check whether synchronization is complete immediately
 *
 * @retval AICL_RET_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
AICL_FUNC_VISIBILITY aiclRet aiclrtSynchronizeStreamWithTimeout(aiclrtStream stream, int32_t timeout);

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_AICL_H_