from multiprocessing import Process, Manager
from subprocess import call

with Manager() as manager:
    file_no = 27
    for index in range(file_no):
        process = Process(target=call, args=(['sh', '../batch/%d.sh' % index],))
        process.start()
