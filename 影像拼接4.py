import cv2
import numpy as np

picture_num=11
picture=np.zeros((picture_num,2160,3840,3),dtype=np.uint8)#62,61,63,60,66,65
picture[0]=cv2.imread('./dataset2/DSC_1160.JPG')#順序:60,67,58,59,68,66,61,62,63,64,65
picture[1]=cv2.imread('./dataset2/DSC_1161.JPG')#反序:68,66,67,58,59,63,60,62,65,64,61
picture[2]=cv2.imread('./dataset2/DSC_1162.JPG')
picture[3]=cv2.imread('./dataset2/DSC_1163.JPG')
picture[4]=cv2.imread('./dataset2/DSC_1164.JPG')
picture[5]=cv2.imread('./dataset2/DSC_1165.JPG')
picture[6]=cv2.imread('./dataset2/DSC_1166.JPG')
picture[7]=cv2.imread('./dataset2/DSC_1167.JPG')
picture[8]=cv2.imread('./dataset2/DSC_1168.JPG')
picture[9]=cv2.imread('./dataset2/DSC_1158.JPG')
picture[10]=cv2.imread('./dataset2/DSC_1159.JPG')
#picture[11]=cv2.imread('./dataset2/DSC_1169.JPG')

kpdetector = cv2.xfeatures2d.SIFT_create() 
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)#特征匹配器

cv2.namedWindow('picture')
pict=np.zeros((picture_num,2160//picture_num,3840//picture_num,3),dtype=np.uint8)
result=np.zeros(((2160//picture_num)*4,(3840//picture_num)*4,3),dtype=np.uint8)
count=np.zeros(((2160//picture_num)*4,(3840//picture_num)*4))
ones=np.ones(((2160//picture_num),(3840//picture_num)))
for i in range(0,picture_num):
    pict[i]=cv2.resize(picture[i],(picture[i].shape[1]//picture_num,picture[i].shape[0]//picture_num))
    #cv2.imshow("picture",pict[i])
out = cv2.VideoWriter('機器視覺HW4-披薩.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
                      ((3840//picture_num)*4,(2160//picture_num)*4))


for i in range(0,picture_num):
    gray = cv2.cvtColor(pict[i], cv2.COLOR_BGR2GRAY)
    kp2 = kpdetector.detect(gray,None)
    dt2 = kpdetector.compute(gray,kp2)[1]
    if i==0:
        T      = np.eye(3)
        T[0,2] = result.shape[1]//2
        T[1,2] = result.shape[1]//8
        
        result = cv2.warpPerspective(pict[i],T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count= cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        count += t_count.astype(np.float)
        disp = result.copy()
        
        cv2.imshow('stitched image',disp.astype(np.uint8))
        
        pict1 = pict[i]
        kp1 = kp2
        dt1 = dt2
    else:
        matches = bf.match(dt2,dt1)#特征匹配
        print('{}, # of matches:{}'.format(i,len(matches)))
        matches = sorted(matches, key = lambda x:x.distance)
        
        src = []
        dst = []
        for m in matches:
            src.append(kp2[m.queryIdx].pt + (1,))
            dst.append(kp1[m.trainIdx].pt + (1,))   
        src = np.array(src,dtype=np.float)
        dst = np.array(dst,dtype=np.float)
        
        # find a homography to map src to dst
        A, mask = cv2.findHomography(src, dst, cv2.RANSAC) 
        # map to the first frame
        T = T.dot(A)
        
        warp_img = cv2.warpPerspective(pict[i],T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count  = cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        
        result+= warp_img
        count += t_count.astype(np.float)

        t_count= count.copy()
        t_count[t_count == 0] = 1
        disp = result.copy()
        
        disp[:,:,0] = result[:,:,0] / t_count
        disp[:,:,1] = result[:,:,1] / t_count
        disp[:,:,2] = result[:,:,2] / t_count
 
        cv2.imshow('matching',cv2.drawMatches(pict[i],kp2,pict1,kp1,matches[:15], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
        cv2.imshow('stitched image',disp.astype(np.uint8))
        out.write(disp.astype(np.uint8))
        
        pict1 = pict[i]
        kp1 = kp2
        dt1 = dt2
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break
cv2.waitKey()
out.release()
cv2.destroyAllWindows()