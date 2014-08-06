"""
Example of using a pool of workers
"""

import sys
import multiprocessing
from multiprocessing import Process, Manager, Pool, Value, Lock


def do_work(args):
    """
    Worker routine
    """
    print args
    entry = args[0]
    queueout = args[1]
    if entry == "DONE":
        # all done
        queueout.put("DONE")
    else:
        # DO SOME WORK

        curLine = "%s\n" %(entry)

        # write the output line to the queueout
        queueout.put(curLine)


def write_from_queue(things_to_write_queue, outfilename, NUM_PARSERS, counter, lock):
    """
    Process in charge of writing the contents of the queue to the output file
    """
    with open(outfilename, "a") as fh:
        while True:
            line = things_to_write_queue.get()
            print 'Test'
            if line == "DONE":
                with lock:
                    counter.value += 1
            if counter == NUM_PARSERS:
                break
            fh.write(line)


def main():
    outputFilename = "output.txt"

    # Set up queues
    jobs = []
    threadManager = Manager()
    outputqueue = threadManager.Queue()

    # Populate the jobs list (below is just a dummy loop)
    for index in range(0,500):
        jobs.append((index, outputqueue)) # Note this is a list of tuples

    # Get the max number of threads
    nthreads = multiprocessing.cpu_count()
    if nthreads == 1:
        print "ERROR ONLY 1 CPU FOUND"
        sys.exit()
    NUM_WORKERS = nthreads - 1  # save a thread for the writer

    # Add a done message to the back of the queue for each of the workers
    for parser in range(NUM_WORKERS):
        jobs.append(("DONE", outputqueue))

    pool = Pool(NUM_WORKERS)

    # Create a single process to write
    counter = Value("i", 0)
    lock = Lock()
    writer = Process(target=write_from_queue, args=(outputqueue, outputFilename, NUM_WORKERS, counter, lock))
    writer.start()

    # Start up the pool of workers
    pool.map(do_work, jobs)

    # This main function will pause execution until the queues are empty
    # If the queues are empty then we are all finished
    pool.close()
    pool.join()
    writer.join()


if __name__ == "__main__":
    main()

