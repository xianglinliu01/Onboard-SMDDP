#include <iostream>
#include <pthread.h>
#include<unistd.h>

using namespace std;

void *thread(void *vargp);
void *thread2(void *vargp);

pthread_mutex_t lock;

int main()
{  // test pthread
    pthread_t tid1;
    pthread_t tid2;

    pthread_create(&tid1, NULL, thread, NULL);
    pthread_create(&tid2, NULL, thread2, NULL);
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
    pthread_mutex_destroy(&lock);
    return 0;
}

void *thread(void *vargp)
{  // NOLINT
    pthread_mutex_lock(&lock);
    cout << "hello" << endl;
    cout << "hello" << endl;
    pthread_mutex_unlock(&lock);
}

void *thread2(void *vargp)
{
    pthread_mutex_lock(&lock);
    cout << "world" << endl;
    cout << "world" << endl;
    pthread_mutex_unlock(&lock);
}