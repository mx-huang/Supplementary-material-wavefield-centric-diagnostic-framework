from  abaqus import*
from  caeModules import *
from  abaqusConstants import*
from  visualization import *
from  odbAccess import *
from  abaqusConstants import *
from  textRepr import *
from  odbSection import *
import sys
sys.path.append(r'E:\program_file\anaconda\envs\abaqus_py\Lib\site-packages')
import time

#批量化提交作业
print("-----开始批量计算-----")
jobs=mdb.jobs.keys()                #获取所有任务名字
for i in jobs:                      #遍历所有任务
    myJob=mdb.jobs[i]               #获取任务对象
    #判断任务是否已经提交，避免重复提交
    if myJob.status == None:
        t0=time.time()              #记录当前提交时间
        myJob.submit()              #提交计算
        myJob.waitForCompletion()   #等待计算完成
        print("【%s】计算完成，耗时 %f 秒"%(i,time.time()-t0))
print("-----计算结束-----")
