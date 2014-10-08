#!/home/wangwei/install/anaconda/bin/python
import sys
import os
import numpy as np

fin=open(sys.argv[1])
nmsg=int(sys.argv[2])
lines=fin.readlines()
niters=len(lines)/nmsg/2
print "iteration num %d" % niters
getary=np.zeros((niters, nmsg))
updateary=np.zeros((niters, nmsg))
l=0
for k in range(niters):
    for i in range(nmsg):
        getary[k,i]=float(lines[l].split(":")[1])
        l+=1
    for i in range(nmsg):
        updateary[k,i]=float(lines[l].split(":")[1])
        l+=1
if l!=len(lines):
    print "error l= %d, lines=%d" % (l, len(lines))

fout=open("perflog.txt", "w")
fout.write("\n niters=%d, nmsg=%d" %(niters, nmsg))
fout.write("\n get avg for niters, %.4f seconds, total latency for all msgs %.4f\n"% (np.average(getary), np.average(np.sum(getary,1))))
np.savetxt(fout, np.average(getary,0), fmt="%.4f", newline="\t")
fout.write("\n get std for niters\n")
np.savetxt(fout, np.std(getary,0), fmt="%.4f", newline="\t")
fout.write("\n update avg for niters, %.4f seconds, total latency for all msgs %.4f\n" % (np.average(updateary), np.average(np.sum(updateary,1))))
np.savetxt(fout, np.average(updateary,0), fmt="%.4f", newline="\t")
fout.write("\n update std for niters\n")
np.savetxt(fout, np.std(updateary,0), fmt="%.4f", newline="\t")

