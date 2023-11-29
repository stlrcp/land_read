/*
// write_shm.c
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

int main()
{
    // 1. 创建一个共享内存
    int shmid = shmget(100, 4096, IPC_CREAT | 0664);
    printf("shmid: %d\n", shmid);

    // 2. 和当前的进程进行关联
    void *ptr = shmat(shmid, NULL, 0);

    // 3. 写数据
    char* str = "hello world";
    memcpy(ptr, str, strlen(str) + 1);

    printf("按任意键继续\n");
    getchar();

    // 4. 解除关联
    shmdt(ptr);

    // // 5. 删除共享内存
    // shmctl(shmid, IPC_RMID, NULL);

    return 0;
}
*/

// read_shm.c
#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

int main()
{
    // 1. 获得一个共享内存
    int shmid = shmget(100, 0, IPC_CREAT);
    printf("shmid:%d\n", shmid);

    // 2. 和当前进程进行关联
    void *ptr = shmat(shmid, NULL, 0);

    // 3. 读数据
    printf("%s\n", (char *)ptr);

    printf("按任意键继续\n");
    getchar();

    // 4. 解锁关联
    shmdt(ptr);

    // 5. 删除共享内存
    shmctl(shmid, IPC_RMID, NULL);

    return 0;
}