from numpy import *;
import numpy as np;
import createMic as Mic;
import os;
import struct;
import cv2;
#from scipy.fftpack import fft,ifft
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from PIL import Image 
#from mpl_toolkits.mplot3d import Axes3D
import time

from PIL import Image

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *

from functools import partial

from multiprocessing import Process,Queue,Pipe

machine = 'SPI'

from GUI import Ui_MainWindow
if machine != 'PC':
    import spidev
import threading
import queue



class B_pro_Thread(threading.Thread):
    def __init__(self,threadID,name,q,TH_Q_SPL):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.q = q
        
    def run(self):
        B_sum = np.array(np.zeros((grid_y,grid_x,3)))
        cnt = [0,0,0];
        while(True):
            if not self.q.empty():
                [B,fram] = self.q.get(True,10);
                B_sum[:,:,fram] = B_sum[:,:,fram] +B.T;
                cnt[fram] += 1;
                if(cnt[fram] == sort_num/2):
                    cnt[fram] = 0;
                    SPL = 20*(np.log10(1+array(B_sum[:,:,fram].real)**0.5/2)+5);
                    B_sum[:,:,fram] = np.array(np.zeros((grid_y,grid_x)))
                    TH_Q_SPL_Lock.acquire();
                    TH_Q_SPL.queue.clear();
                    TH_Q_SPL.put(SPL)
                    TH_Q_SPL_Lock.release();

                    # B_sum = np.zeros((grid_y,grid_x))
            else:
                pass




class mywindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self,q):
        super(mywindow,self).__init__()
        
        # 摄像头
        self.CAM_NUM = 0
        self.cap = cv2.VideoCapture() #初始化摄像头
        self.timerCamera = QTimer() #初始化定时器
        
        # UI
        self.setupUi(self)
        desktop_geometry = QtWidgets.QApplication.desktop()           #获取屏幕大小
        main_window_width= desktop_geometry.width()                    #屏幕宽
        main_window_height = desktop_geometry.height()                 #屏幕高
        rect = self.geometry()                                         #获取窗口界面大小
        window_width = rect.width() #窗口界面的宽
        window_height = rect.height() #窗口界面的高
        x = (main_window_width - window_width) // 2 #计算窗口左上角点横坐标
        y = (main_window_height - window_height) //2#计算窗口左上角纵坐标

        self.setGeometry(x, y, window_width, window_height)

        # self.image_label.setScaledContents(True)    #让图片适应QLabel
        self.B1.setText('Open Camera')

        self.q = q;

        self.image_label.setGeometry(80, 60, 640, 480)

        self.init_imag();
        
        # self.mult_par = 32;

        # self.image_height   =   grid_x * self.mult_par;
        # self.image_width    =   grid_y * self.mult_par;
        self.image_height   =   grid_x * mult_par;
        self.image_width    =   grid_y * mult_par;
        self.image_depth    =   3
        self.B3 = np.array(np.zeros((self.image_height,self.image_width,self.image_depth)))
        
        self.final_im2 = np.array(np.zeros((self.image_height,self.image_width,self.image_depth)));

        # 槽连接
        # self.timerCamera.timeout.connect(self.showCamera)
        self.timerCamera.timeout.connect(self.showImage)
        self.B1.clicked.connect(self.cameraSwitch)
        # self.B1.clicked.connect(self.showImage)
        self.B2.clicked.connect(self.closeWindow)

        self.Bsave.clicked.connect(self.saveImage)

        # self.B3 = np.array(np.zeros((grid_y,grid_x)));
        self.setWindowFlags(Qt.FramelessWindowHint)# 无边框
        # self.setAttribute(Qt.WA_TranslucentBackground) #背景透明

        # self.Im = cv2.imread('dog.jpg')
        self.Im = cv2.imread('school.jpg')


    def saveImage(self):
        im=cv2.cvtColor(self.final_im2, cv2.COLOR_RGB2BGR)
        file_name = time.ctime().replace(' ','_').replace(':','_')+'.jpg';
        cv2.imwrite(file_name,im);

    def showImage(self):
        # print('showImage');
        # SPL =np.array(np.zeros((grid_y,grid_x)))

        if not self.q.empty():
            B_Temp = self.q.get()
            B_Temp = cv2.flip(B_Temp,-1)
            self.B3[:,:,0] = B_Temp
            self.image_height,self.image_width,self.image_depth = self.B3.shape ;    # 获取图像的高，宽以及深度。
        
        flag,cap_im = self.cap.read();
        if flag==0:
            print("false");
            return;
        cap_height,cap_width,cap_depth = cap_im.shape ; 
        off_height = round((cap_height - self.image_height)/2);
        off_width = round((cap_width - self.image_width)/2);

        cap_im2 = cv2.cvtColor(cap_im[off_height:self.image_height+off_height,off_width:self.image_width+off_width,0:self.image_depth], cv2.COLOR_BGR2RGB);       # opencv读图片是BGR，qt显示要RGB，
        # final_im = self.B3*0.3 + cap_im2*0.7
        
        final_im = cv2.add(uint8(self.B3),uint8(cap_im2))
        self.final_im2 = uint8(final_im)
        QIm = QImage(self.final_im2.data,self.image_width,self.image_height,       # 创建QImage格式的图像，并读入图像信息
        self.image_width * self.image_depth,
        QImage.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(QIm))

        
        # maxSPL = 1




    def showCamera(self):
        flag,Im = self.cap.read();
        image_height,image_width,image_depth = Im.shape ;    # 获取图像的高，宽以及深度。
        QIm = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB);       # opencv读图片是BGR，qt显示要RGB，

        QIm = QImage(QIm.data,image_width,image_height,       # 创建QImage格式的图像，并读入图像信息
                        image_width * image_depth,
                        QImage.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(QIm))

    def closeWindow(self):
        if self.timerCamera.isActive() == True:
            # self.closeCamera()
            self.closeDAS()
            # sys.exit(0);
        self.close()

    def cameraSwitch(self):
        if self.timerCamera.isActive() == False:
            self.startDAS();
            # self.startCamera()
        else:
            self.closeDAS();
            # self.closeCamera();

    def startDAS(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            mas = QMessageBox.Warning(self,u'Warning',u'Plz check camera!',
                buttons = QMessageBox.Ok,
                defaultButton = QMessageBox.Ok)
        else:
            self.timerCamera.start(20)
            self.B1.setText('Close DAS')
        # self.timerCamera.start(40)
        # self.B1.setText('Close DAS')


    def closeDAS(self):
        self.timerCamera.stop();
        self.image_label.clear()
        self.B1.setText('Open DAS')
        self.init_imag()

        self.cap.release()




    def DASSwitch(self):
        if self.timerCamera.isActive() == False:
            self.startDAS();
        else:
            self.closeDAS();


    
    def startCamera(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            mas = QMessageBox.Warning(self,u'Warning',u'Plz check camera!',
                buttons = QMessageBox.Ok,
                defaultButton = QMessageBox.Ok)
        else:
            self.timerCamera.start(10)
            self.B1.setText('Close Camera')

    def closeCamera(self):
        self.timerCamera.stop();
        self.cap.release()
        self.image_label.clear()
        self.B1.setText('Open Camera')
        self.init_imag()
    

    def init_imag(self):
        # Im = cv2.imread('dog.jpg')
        Im = cv2.imread('school.jpg')

        image_height,image_width,image_depth = Im.shape     # 获取图像的高，宽以及深度。
        QIm = cv2.cvtColor(Im, cv2.COLOR_BGR2RGB)       # opencv读图片是BGR，qt显示要RGB，

        QIm = QImage(QIm.data,image_width,image_height,       # 创建QImage格式的图像，并读入图像信息
                        image_width * image_depth,
                        QImage.Format_RGB888)

        self.image_label.setPixmap(QPixmap.fromImage(QIm))









def steerVector(plane_distance, frequencies, scan_limits, grid_resolution, mic_positions, c):
    print('\t------------------------------------------\n');
    print('\tStart calculating steering vector...\n');

    N_mic = size(mic_positions, 1);
    N_freqs = len(frequencies);

    X = np.arange(scan_limits[0][0],scan_limits[0][1]+grid_resolution,grid_resolution);
    Y = np.arange(scan_limits[1][0],scan_limits[1][1]+grid_resolution,grid_resolution);
    Z = plane_distance;
    N_X = len(X);
    N_Y = len(Y);
    N_Z = 1;

    N_scanpoints = np.dot(N_X,N_Y);
    N_scanpoints = np.dot(N_scanpoints,N_Z);

    x_t = np.mat(np.zeros((N_scanpoints,3)));

    x_t[:, 0] = tile(X, N_Y).reshape(N_Y*N_X ,1);
    x_t[:, 1] = tile(Y, N_X).reshape((N_Y,N_X),order='C').reshape((N_Y*N_X,1),order = 'F')
    x_t[:, 2] = plane_distance;

    x_0 = mean(mic_positions, 1);

    r_t0 = array( array((x_t[:,0] - x_0[0]))**2 + array((x_t[:,1] - x_0[1]))**2 + array(x_t[:,2] - x_0[2])**2 )**0.5;
    # h = np.mat(np.zeros((N_mic,size(x_t, 0),N_freqs),dtype(complex)));
    h = np.array(np.zeros((N_mic,size(x_t, 0),N_freqs),dtype(complex)));
    for K in range(0,N_freqs):
        k = 2*pi*frequencies[K]/c;
        sum_r_ti = np.array(np.zeros((N_scanpoints, 1)),dtype(float));

        for I in range(0,N_mic):
            r_ti = array( array(x_t[:,0] - mic_positions[0,I])**2 + \
                array(x_t[:,1] - mic_positions[1,I])**2 + \
                array(x_t[:,2] - mic_positions[2,I])**2 )**0.5;
            sum_r_ti = sum_r_ti + array(r_ti)**(-2);
            h[I, :, K] = (exp(-1j*k*(r_ti-r_t0))/(r_ti*r_t0))[:,0];

        for I in range(0,N_mic):
            h[I, :, K] = h[I, :, K] / sum_r_ti.T;

        # h_conj = h;
        h_conj = np.conj(h);
    print('\tFinished calculating steering vector!\n');
    print('\t------------------------------------------\n');
    return [h,h_conj,N_mic,x_t]

def DAS(CSM, h, frequencies, scan_limits, grid_resolution):
    # print('\t------------------------------------------\n');
    # print('\tStart beamforming, DAS...\n');

    N_freqs = len(frequencies);
    X = np.arange(scan_limits[0][0],scan_limits[0][1]+grid_resolution,grid_resolution);
    Y = np.arange(scan_limits[1][0],scan_limits[1][1]+grid_resolution,grid_resolution);
    N_X = len(X);
    N_Y = len(Y);
    # print('\tBeamforming for %d frequency points...\n'%(N_freqs));
    
    B = np.array(np.zeros((1,N_X*N_Y)),dtype=complex);
    
    # start =time.time_ns();
    
    for K in range(0,N_freqs):
    # for K in frequencies:
        B = B + sum(h[:,:,frequencies[K]]*np.dot(CSM[:,:,K],np.conj(h[:,:,frequencies[K]])), axis=0, keepdims=True); 
    
    # end = time.time_ns()
    # print('Running time: %s ms'%((end-start)/1000000))
    
    B = B.reshape(N_X, N_Y);

    # print('\tBeamforming complete!\n');
    # print('\t------------------------------------------\n');
    return [X,Y,B]


def DAS_Process_D(q,q2,h,h_conj, scan_limits, grid_resolution,id):
    X = np.arange(scan_limits[0][0],scan_limits[0][1]+grid_resolution,grid_resolution);
    Y = np.arange(scan_limits[1][0],scan_limits[1][1]+grid_resolution,grid_resolution);
    N_X = len(X);
    N_Y = len(Y);
    B = np.array(np.zeros((1,N_X*N_Y)),dtype=complex);

    while(True):
        start =time.time_ns();
        try:
            # [temp,frequencies] = q.get(True,2);
            [CSM,frequencies,data_id] = q.get(True,10);
        except Queue.Empty:
            print('队列空')
            break

        N_freqs = len(frequencies);

        # N_freqs = len(frequencies);
        

        for K in range(0,N_freqs):
        # for K in frequencies:
            B = B + sum(h[:,:,frequencies[K]]*np.dot(CSM[:,:,K],np.conj(h[:,:,frequencies[K]])), axis=0, keepdims=True); 


        # B = sum(np.multiply(h[:,:,frequencies],np.dot(CSM,h_conj[:,:,frequencies])), axis=0); 


        # B = sum(np.multiply(h[:,:,frequencies],np.dot(CSM,h_conj[:,:,frequencies])), axis=0, keepdims=True); 




        # for K in range(0,N_freqs):
            # B = B + sum(h[:,:,frequencies[K]]*np.dot(CSM[:,:,K],np.conj(h[:,:,frequencies[K]])), axis=0, keepdims=True); 
            # B = B + sum(h[:,:,frequencies[K]]*temp[:,:,K],axis=0, keepdims=True); 

        B = B.reshape(N_X, N_Y);

        try:
            q2.put([B,data_id])
        except Queue.full:
            print('B队列满')
            break

        end = time.time_ns()
        print('\t%sDAS F:%s compelete!Running time: %s ms'%(id,data_id,(end-start)/1000000));
        B = np.array(np.zeros((1,N_X*N_Y)),dtype=complex);
        # print('\t------------------------------------------\n');
        # return [X,Y,B]


def data_pre(q,h,h_conj,freq_l,freq_u,Fs,sort_num,N_mic,x_t):
    N_start = 0;
    N_end = 256;
    block_samples = 256;
    x_fr = Fs / block_samples * np.arange(0,int(floor(block_samples/2)-1),1);
    freq_sels_r = np.where((x_fr>=freq_l)*(x_fr<=freq_u));
    N_freqs = sort_num;
    CSM = np.array(np.zeros((64, 64, N_freqs)),dtype=complex);    
    
    # CSM_1 = np.array(np.ones((64, 64)));
    # CSM_1[np.eye(64,dtype=np.bool)] = 0;    
    # CSM = np.array(np.zeros((64, 64)),dtype=complex);    

    Temp =np.array(np.zeros((N_mic,size(x_t, 0),N_freqs),dtype(complex)));
    data_id = 0;
    data1 = []
    if machine != 'PC':
        spi = spidev.SpiDev()
        spi.open(0,0)
        spi.max_speed_hz = 20000000

    while(True):
        start =time.time_ns();
    #################################数据读取##########################
        # print("\tdata_read_start!")
        # start = time.time_ns()
        if machine == 'PC':
            time.sleep(0.027)
            fid = open("mems_data.bin","rb");
            data1 = struct.unpack('f'*64*256*1,fid.read(64*256*4*1))
            fid.close();
        else:
            print("SPI")
            spi.writebytes([0x02])
            a=np.array(spi.readbytes(65536),dtype = np.int8)
            # data1 = np.fromstring(a,dtype=np.float32)
            data1 = np.frombuffer(a,dtype=np.float32)


    #######################CSM##############################

        data2 = data1[64*256*0:64*256*1]
        p = np.asarray(data2).reshape((256,64),order='F');

        # p = data;

        p = np.mat(p);

        p_fft = 2*np.fft.fft(p[np.arange(N_start ,N_end ,1) ,:].T)/block_samples;

        # freq_sels_r = np.where((x_fr>=freq_l)*(x_fr<=freq_u));

        abs_fft = abs(p_fft[0,freq_sels_r])

        sort_abs_fft= np.argsort(-1*abs_fft) + freq_sels_r[0][0]

        freq_sels = sort_abs_fft[0][0:sort_num]

        p_fft = np.mat(p_fft)
        q_fft = p_fft.T
        # start =time.time_ns();

        for F in np.arange(0 , N_freqs,1):
            CSM[:,:,F] = np.dot(q_fft[freq_sels[F],:].H,q_fft[freq_sels[F],:]);
            # CSM[np.eye(64,dtype=np.bool),F] =0;

            # CSM=0.5 * np.dot(q_fft[freq_sels[F],:].H,q_fft[freq_sels[F],:]);


        for i in np.arange(0,int(N_freqs/2),1):
            try:
                # q.put([CSM,freq_sels])
                q.put([CSM[:,:,0+2*i:2*i+2],freq_sels[0+2*i:2*i+2],data_id])
                # q_fram.put(data_id);
            except Queue.full:
                print('队列满')
                break




        end =time.time_ns();
        # CSM = np.array(np.zeros((64, 64)),dtype=complex);    
        CSM = np.array(np.zeros((64, 64, N_freqs)),dtype=complex);    

        if (data_id>1):
            data_id = 0;
        else:
            data_id +=1;
        print("\tdata_pre compelete!Running time: %s ms"%((end-start)/1000000))


def data_rev(q,TH_Q_SPL,grid_y,grid_x,sort_num):
    B_sum = np.array(np.zeros((grid_y,grid_x,3)))
    cnt = [0,0,0];
    while(True):
        if not q.empty():
            [B,fram] = q.get(True,10);
            B_sum[:,:,fram] = B_sum[:,:,fram] +B.T;
            cnt[fram] += 1;
            if(cnt[fram] == sort_num/2):
                cnt[fram] = 0;
                SPL = 200*(np.log10(1+array(B_sum[:,:,fram].real)**0.5/2));
                B_sum[:,:,fram] = np.array(np.zeros((grid_y,grid_x)))
                
                # print ("%s processing %s" % (threadName, data))
                maxSPL = ceil(np.max(SPL));
                minSPL = floor(np.min(SPL))
                # if (maxSPL==100):
                #     maxSPL = minSPL + 1;
                    
                
                print("QT maxSPL=%f,minSPL=%f"%(maxSPL,minSPL))

                BF = 1
                
                list1 = list(np.where(SPL==(maxSPL - BF)));
                print(*list1[1:],sep=' ')

                if(maxSPL-minSPL<6):
                    return
                    

                B2=np.kron(SPL,np.ones((mult_par,mult_par)))
                x2 = len(B2);
                y2 = len(B2[0,:])
                B21 = B2;

                B21[np.where(B21<maxSPL - BF)] =BF;
                
                

                B3 = np.round((B2[:,:]-BF)*255/(maxSPL-BF))
                B3 = uint8(B3)
                if not TH_Q_SPL.full() :
                    TH_Q_SPL.put(B3)
                    time.sleep(0.010)

                    

        else:
            pass









######################################全局变量################################
# TH_Q_SPL_Lock = threading.Lock()
scan_x = [-0.35,0.35];
scan_y = [-0.35,0.35];
mult_par = 32;
scan_resolution = 1/20;
grid_x = round((scan_x[1]-scan_x[0])/scan_resolution) + 1;
grid_y = round((scan_y[1]-scan_y[0])/scan_resolution) + 1;
######################################主程序#################################
if __name__ == '__main__':
    print("\t主进程启动")

    #spi初始化
    # spi = spidev.SpiDev()
    # spi.open(0,0)
    # spi.max_speed_hz = 20000000

#################算法参数设定###################################
    mic_x = [-0.1,0.1];
    mic_y = [-0.1,0.1];

    

    z_source = 1;

    c = 343;  

    # Fs = 31250;
    Fs = 125000;

    sort_num = 4;

    search_freql = 1700;
    # search_frequ = 13000;
    # search_frequ = search_freql + 5000;
    search_frequ = 30000;

##################################权重矩阵计算###############################################
    mic_pos =np.mat(np.zeros((64,3)));
    mic_pos[:,0] = Mic.xcfg[:,0];
    mic_pos[:,1] = Mic.ycfg[:,0];

    start =time.time_ns()
    freqs = Fs / 256 * np.arange(0,int(floor(256/2)-1),1);
    [h,h_conj,N_mic,x_t] = steerVector(z_source, freqs, [scan_x,scan_y], scan_resolution, mic_pos.T, c);
    end = time.time_ns()
    print('Running time: %s ms'%((end-start)/1000000))
#########################################队列定义#############################################

    #************************************进程队列****************************************#
    q2 = Queue(3)
    q = Queue(3)     
    TH_Q_SPL = Queue(2)
    #************************************线程锁、队列****************************************#

    # TH_Q_SPL = queue.Queue(2)

#########################################进程定义##################################################

    #******************************数据采集进程*******************************************#
    p2 = Process(target=data_pre,args=(q,h,h_conj,search_freql,search_frequ,Fs,sort_num,N_mic,x_t,));
    p2.daemon=True
    # p2 = Process(target=DAS_Process,args=(parent_con,h, [scan_x,scan_y], scan_resolution))

    #******************************计算进程*********************************************#
    p3 = Process(target=DAS_Process_D,args=(q,q2,h,h_conj, [scan_x,scan_y], scan_resolution,1,));
    p3.daemon=True
    p4 = Process(target=DAS_Process_D,args=(q,q2,h,h_conj, [scan_x,scan_y], scan_resolution,2,));
    p4.daemon=True
    # p5 = Process(target=DAS_Process_D,args=(q,q2,h,h_conj, [scan_x,scan_y], scan_resolution,3,));
    # p5.daemon=True



    PdataRev =Process(target = data_rev,args = (q2,TH_Q_SPL,grid_y,grid_x,sort_num))
    PdataRev.daemon=True



    

    # p6 = Process(target=DAS_Process_D,args=(q,q2,h,h_conj, [scan_x,scan_y], scan_resolution,3,));
##########################################主进程线程定义##############################################

    # thread1 = myThread(1, "Thread-1",TH_Q_SPL)
    # thread3 = B_pro_Thread(2, "Thread-2",q2,TH_Q_SPL)

    # thread3.setDaemon(True)
    # app = QtWidgets.QApplication(sys.argv)
    # window = mywindow(TH_Q_SPL)
    # window.show()
##########################################启动线程################################################
    # thread1.start()
    # thread3.start()
##########################################启动进程###############################################
    start =time.time_ns();



    ####接收进程
    # p1.start();
    p2.start();
    # p2.start();

    ####计算进程
    p3.start();
    p4.start();
    # p5.start();

    ####SPL进程
    PdataRev.start();



    # p6.start();

##########################################主进程处理############################################
    
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow(TH_Q_SPL)
    window.show()
    end =time.time_ns();
    print("\t1round!Running time: %s ms"%((end-start)/1000000))
    sys.exit(app.exec_())

    print("\t主进程退出")


#测试git->vscode


