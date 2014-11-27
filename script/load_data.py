#!/usr/bin/env python

import sys
import os
import subprocess
import argparse
import time
from time import sleep
from random import shuffle

def partition(records, workers, m, sharddir, replicate):
    '''
    partition records and copy record id file to workers
    m: group size
    '''
    ngroups=(len(workers)/m)
    nGroupRecords=len(records)/ngroups
    for g in range(ngroups):
        groupRecords=records[g*nGroupRecords:(g+1)*nGroupRecords]
        if replicate:
            with open("tmp", "w") as fd:
                for record in groupRecords:
                    fd.write(record)
                fd.flush()
            for w in workers[g*m:g*m+m]:
                # write records to file & copy to worker's shard folder
                cmd='ssh %s "mkdir -p %s" && scp tmp %s:%s/rid.txt && ssh %s "chmod 444 %s/rid.txt"' \
                    %(w, sharddir, w, sharddir, w, sharddir)
                print cmd
                os.system(cmd)
        else:
            nWorkerRecords=nGroupRecords/m
            for (k,w) in enumerate(workers[g*m:g*m+m]):
                workerRecords=groupRecords[k*nWorkerRecords:(k+1)*nWorkerRecords]
                with open("tmp", "w") as fd:
                    for record in workerRecords:
                        fd.write(record)
                    fd.flush()
                # write records to file& copy to worker's shard folder
                cmd='ssh %s "mkdir -p %s" && scp tmp %s:%s/rid.txt && ssh %s "chmod 444 %s/rid.txt"' \
                    %(w, sharddir, w, sharddir, w, sharddir)
                print cmd
                os.system(cmd)


def download(hdfs_folder, local_folder, workers, exe_folder):
    '''
    download images from hdfs
    local_folder: local shard folder
    exe_folder: download is conducted by download.sh script in this folder
    '''
    for worker in workers:
        cmd='ssh %s "%s/download.sh %s/rid.txt %s %s/img" &' %(worker, exe_folder, local_folder, hdfs_folder, local_folder)
        print cmd
        os.system(cmd)

    print '***********************************************************'
    print '**************************Progress*************************'
    print '***********************************************************'
    try:
        nalive=len(workers)
        while nalive>0:
            for worker in workers:
                sys.stdout.write('%s---' % worker)
                sys.stdout.flush()
                cmd='ssh %s "ls %s/img |wc -l && cat %s/rid.txt |wc -l"' % (worker, local_folder,local_folder)
                outputs = subprocess.check_output(cmd, shell=True).split('\n')
                total=int(outputs[1])
                cur=int(outputs[0])
                print 'downloaded %5d, total %5d, percentage %.3f%%' %(cur, total, cur*100.0/total)

                #if finish download , then minus num of alive workers
                cmd='ssh %s "if ps ax |grep download.sh |grep -v ssh|grep -v grep ; then echo OK ; else echo NO ; fi"' % worker
                output = subprocess.check_output(cmd, shell=True)
                if 'NO' in output:
                    nalive-=1
            sleep(15)
            print '******************%s***********************' % time.strftime("%c")
    except KeyboardInterrupt:
        for worker in workers:
            os.system('ssh %s "ps ax|pgrep download.sh|xargs kill"' %worker)
        sys.exit()

def create_shard(hostfile, local_folder, workers, resize, mfile):
    '''
    run mpi job to parse images and insert records to shard.dat
    mfile: mean file path
    '''
    cmd='mpirun -np %d -hostfile %s ./loader --dir=%s --mean=%s --width=%d --height=%d ' %(len(workers), hostfile, local_folder, mfile, resize, resize)
    print cmd
    os.system(cmd)
    sys.exit()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Partition dataset by assgin each worker a image path list; Load Data from HDFS by hadoop fs get; Create Local Shards by loader.')
    parser.add_argument('n',type=int, default=1000, help='dataset size')
    parser.add_argument('-m', type=int, default=1, help='group size')
    parser.add_argument('-r', type=bool, default=True, help='replciate data within one group')
    parser.add_argument('-s', type=int, default=256, help='resize image')
    parser.add_argument('--phase', choices=['train', 'validation', 'test'], help="dataset name/phase")
    parser.add_argument('--mfile', default='examples/imagenet12/imagenet_mean.binaryproto', help='image mean')
    parser.add_argument('--rfile', default='examples/imagenet12/validation_label.txt', help='record list file which may have mroe records than n')
    parser.add_argument('--host', default='examples/imagenet12/hostfile', help='record list file which may have mroe records than n')
    parser.add_argument('--local', default='/data1/wangwei/lapis/', help='local data dir, root dir for storing data of SINGA, must be an absolute path')
    parser.add_argument('--hdfs', default='imagenet12/validation', help='hdfs dir')
    args=parser.parse_args();
    if not os.path.isabs(args.local):
        print 'local root folder %s is not a absolute path' % args.local
    local_folder=os.path.join(args.local, args.phase)
    with open(args.rfile) as fd:
        records=fd.readlines()
    with open(args.host) as fd:
        workers=[worker.strip() for worker in fd.readlines()]
    shuffle(records)
    records=records[0:args.n]

    partition(records, workers, args.m, local_folder, args.r)

    #absolute path of script folder
    if os.path.isabs(sys.argv[0]):
        exedir=os.path.dirname(sys.argv[0])
    else:
        exedir=os.path.join(os.getcwd(),os.path.dirname(sys.argv[0]))
    download(args.hdfs, local_folder, workers, exedir)

    if 'loader' not in os.listdir(os.getcwd()):
        print 'loader exec cannot be found in the current working directory'
    else:
        create_shard(args.host, local_folder, workers, args.s, args.mfile)
