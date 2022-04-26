import multiprocessing
import threading


def test_import_thread_mp():
    def f1():
        print(f' f1 start import ')
        import docarray

        print(f' f1 imported {docarray.__version__}')

    def f2():
        print(f' f2 start import ')
        import docarray

        print(f' f2 imported {docarray.__version__}')

    x1 = threading.Thread(target=f1)
    x2 = multiprocessing.Process(target=f2)
    x2.start()
    x1.start()
    x2.join()
    x1.join()
