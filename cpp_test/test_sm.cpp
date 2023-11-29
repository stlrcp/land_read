//  C++ 共享内存 基础案例
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <stdio.h>

int main()
{
    int shmid;  // 共享内存标识符
    printf("pid = %d\n", getpid());

    // 创建共享内存，键值为0X5005，共1024字节
    if((shmid = shmget((key_t)0x5005, 1024, 0640|IPC_CREAT)) == -1)
    {
        printf("shmat(0x5005) failed \n");
        return -1;
    }

    char *ptext = 0;  // 用于指向共享内存的指针

    // 将共享内存连接到当前进程的地址空间，由ptext指针指向它
    ptext = (char *)shmat(shmid, 0, 0);

    // 操作本程序的ptext指针，就是操作共享内存
    printf("写入前：%s\n", ptext);
    sprintf(ptext, "本程序的进程号是：%d", getpid());
    printf("写入后：%s\n", ptext);

    // 把共享内存从当前进程中分离
    shmdt(ptext);

    // 删除共享内存
    // if (shmctl(shmid, IPC_RMID, 0) == -1)
    // {
    //     printf("shmctl(0x5005) failed\n");
    //     return -1;
    // }
}
// ipcs -m     查看共享内存
// ipcrm -m $shmid   删除共享内存shmid