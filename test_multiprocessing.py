# file to test multiprocessing
import multiprocessing
import time
import os

def worker():

    d = dict()
    l = list()
    print(f"开始子进程{os.getpid()}") 
    with multiprocessing.Lock():  # 手动加锁
        for i in range(10000):
            name = f'count_{i}'
            d[name] = d.get(name, 0) + 1
        l.append(multiprocessing.current_process().name)
    
    print(f"结束子进程{os.getpid()}") 

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        # 创建共享字典和列表
        #shared_dict = manager.dict()
        #shared_list = manager.list()
        
        # 创建多个进程
        #processes = [multiprocessing.Process(target=worker, args=(shared_dict, shared_list)) for _ in range(1024)]
        processes = [multiprocessing.Process(target=worker, args=()) for _ in range(10240)]
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        
        #print(f"字典结果: {shared_dict}")  # 输出: {'count': 5}
        #print(f"列表结果: {shared_list}")  # 输出: 5个进程名