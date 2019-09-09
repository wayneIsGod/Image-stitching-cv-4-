import cv2
import numpy as np

cap    = cv2.VideoCapture('hw4_dataset1.mp4')
print('height:{} width:{}'.format(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter('機器視覺HW4-橋2.avi',cv2.VideoWriter_fourcc(*'XVID'), 20.0, 
                      ((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4*3,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4)))

def set_frame_number(x):
    global frame_num
    frame_num = x
    return
result = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4*3,3))#>>> 9//2 4  >>> -9//2  -5
count  = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4*3))
ones = np.ones(((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))//4,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))//4)))

cv2.namedWindow('matching')
cv2.createTrackbar('frame no.','matching',0,total_frame-1,set_frame_number)

kpdetector = cv2.xfeatures2d.SIFT_create() 
#kpdetector = cv2.AKAZE_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
start_frame_num = total_frame//2-12
frame_num = start_frame_num

while frame_num < total_frame and frame_num > 0:
    cv2.setTrackbarPos('frame no.','matching',frame_num)
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
    ret, frame2 = cap.read() 
    if ret==False:
        break    
    
    frame2 = cv2.resize(frame2,(frame2.shape[1]//4,frame2.shape[0]//4))
    #print(frame2,(frame2.shape[1]//4,frame2.shape[0]//4))
    
    #kp1, dt1 = kpdetector.detectAndCompute(frame1,None)
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    kp2 = kpdetector.detect(gray,None)
    dt2 = kpdetector.compute(gray,kp2)[1]
    
    if frame_num == start_frame_num:
        print(result.shape[1],frame2.shape[1])
        T      = np.eye(3)
        T[0,2] = ((result.shape[1]-(frame2.shape[1]))//2)-50
        T[1,2] = 0
        result = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count= cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)#每個連續畫面貼一次，最後內插用
        count += t_count.astype(np.float)
        disp = result.copy()
        cv2.imshow('stitched image',disp.astype(np.uint8))
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
        frame_num += 1
        
    elif frame_num >= start_frame_num:
        if frame_num==total_frame-1:
            frame_num=start_frame_num-1
            continue
        # Match descriptors.
        matches = bf.match(dt2,dt1)
        print('{}, # of matches:{}'.format(frame_num,len(matches)))

        # Sort in ascending order of distance.
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
        warp_img = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count  = cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        result+= warp_img
        count += t_count.astype(np.float)

        t_count= count.copy()
        t_count[t_count == 0] = 1
        disp = result.copy()
        
        disp[:,:,0] = result[:,:,0] / t_count
        disp[:,:,1] = result[:,:,1] / t_count
        disp[:,:,2] = result[:,:,2] / t_count
 
        cv2.imshow('matching',cv2.drawMatches(frame2,kp2,frame1,kp1,matches[:15], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
        cv2.imshow('stitched image',disp.astype(np.uint8))
        out.write(disp.astype(np.uint8))
        
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
        frame_num += 1
    else:
        # Match descriptors.
        matches = bf.match(dt2,dt1)
        print('{}, # of matches:{}'.format(frame_num,len(matches)))

        # Sort in ascending order of distance.
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
        warp_img = cv2.warpPerspective(frame2,T,(result.shape[1],result.shape[0])).astype(np.float)
        t_count  = cv2.warpPerspective(ones,T,(result.shape[1],result.shape[0])).astype(np.float)
        result+= warp_img
        count += t_count.astype(np.float)

        t_count= count.copy()
        t_count[t_count == 0] = 1
        disp = result.copy()
        
        disp[:,:,0] = result[:,:,0] / t_count
        disp[:,:,1] = result[:,:,1] / t_count
        disp[:,:,2] = result[:,:,2] / t_count
 
        cv2.imshow('matching',cv2.drawMatches(frame2,kp2,frame1,kp1,matches[:15], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS))
        cv2.imshow('stitched image',disp.astype(np.uint8))
        out.write(disp.astype(np.uint8))
        
        frame1 = frame2
        kp1 = kp2
        dt1 = dt2
        frame_num -= 1
    key = cv2.waitKey(20) & 0xFF
    if key == 27:
        break

cv2.waitKey()    
cap.release()
out.release()
cv2.destroyAllWindows()