from multiprocessing import Process, Pipe, Lock, Value, cpu_count
from threading import Thread
from queue import PriorityQueue
import time


class TerminateSignal():
    def __init__(self):
        pass


def worker(input_pipe, input_lock, output_pipe, output_lock, error_value, function, args, kwargs):
    try:
        while True:
            with input_lock:
                inp = input_pipe.recv()
                if type(inp) == TerminateSignal:
                    return
                idx, item = inp

            result = function(item, *args, **kwargs)

            with output_lock:
                output_pipe.send((idx, result))
    except Exception as e:
        import traceback
        import sys
        traceback.print_exception(*sys.exc_info())
        error_value.value = 1


def consumer(input_pipe, last_finished_idx, error_value, function, args, kwargs):
    try:
        buffer = PriorityQueue()
        next_idx = 0
        while True:
            inp = input_pipe.recv()
            if type(inp) == TerminateSignal:
                return
            idx, item = inp
            if idx != next_idx:
                buffer.put((idx, item))
                continue

            while True:
                function(item, *args, **kwargs)
                last_finished_idx.value = idx
                next_idx += 1

                if buffer.empty():
                    break

                idx, item = buffer.get()
                if idx != next_idx:
                    buffer.put((idx, item))
                    break
    except Exception as e:
        import traceback
        import sys
        traceback.print_exception(*sys.exc_info())
        error_value.value = 1


def server(worker_function, consumer_function, inputs, num_threads=int(cpu_count()/2), worker_process_name='Worker_Process_{}', consumer_process_name='Consumer_Process', use_process=True, *args, **kwargs):

    if num_threads == 1:
        for inp in inputs:
            res = worker_function(inp, *args, **kwargs)
            consumer_function(res, *args, **kwargs)
        return

    task_pipe = Pipe()
    result_pipe = Pipe()
    last_finished_idx = Value('i', 0)
    error_value = Value('i', 0)
    lock1 = Lock()
    lock2 = Lock()

    if use_process:
        proc_type = Process
    else:
        proc_type = Thread

    worker_processes = [proc_type(target=worker, args=(task_pipe[1], lock1, result_pipe[0], lock2, error_value, worker_function, args, kwargs), name=worker_process_name.format(i)) for i in range(num_threads)]
    consumer_process = proc_type(target=consumer, args=(result_pipe[1], last_finished_idx, error_value, consumer_function, args, kwargs), name=consumer_process_name)

    try:
        for p in worker_processes:
            p.start()
        consumer_process.start()

        if error_value.value > 0:
            raise BaseException('Exception in one of the processes!')

        last = None
        for i, inp in enumerate(inputs):
            last = i
            task_pipe[0].send((i, inp))
            if error_value.value > 0:
                raise BaseException('Exception in one of the processes!')
        for _ in range(num_threads):
            task_pipe[0].send(TerminateSignal())

        while last_finished_idx.value != last:
            if error_value.value > 0:
                raise BaseException('Exception in one of the processes!')
            time.sleep(1)

    finally:
        for p in worker_processes:
            p.join()
        result_pipe[0].send(TerminateSignal())
        consumer_process.join()


def print_function(x):
    print(x)
