from __future__ import division
import sys
import numpy as np
import cv2
class Color():
    RED     = (0,0,255)
    GREEN   = (0,255,0)
    BLUE    = (255,0,0)
    YELLOW  = (0,255,255)
    CYAN    = (255,255,0)
    MAGENTA = (255,0,255)

MIN_PIX=60 # pixel threshold
def nothing():
    pass

#it is a class because we want it to wrok with ros image streams and video
class SimpleBoundingBox(object):
    #indexing is better for option comparisons
    CIRCLE=0
    RECTANGLE=1

    MOG = 0
    MOG2 = 1
    GMG = 2
    MAN = 3
    WTR = 4


    BG_METHODS=[cv2.bgsegm.createBackgroundSubtractorMOG,
     cv2.createBackgroundSubtractorMOG2,
     cv2.bgsegm.createBackgroundSubtractorGMG, nothing, nothing]
    def __init__(self,method=MAN, bg=[]):
        self.subtractor=SimpleBoundingBox.BG_METHODS[method]()
        self.set_frames()
        self.diff_thresh=55
        self.results=[0,0,0]
        self.N=0
        self._training = 0
        if method == SimpleBoundingBox.GMG:
            self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            self.subt = self.subt_GMG

        elif method == SimpleBoundingBox.WTR:
            self.subt = self.subt_watershed
            self.kernel1 = np.ones((3,3),np.uint8)
            self.kernel2 = np.ones((11,11),np.uint8)

        elif method == SimpleBoundingBox.MAN:
            #if bg: self.capture_bg(bg)
            self.subt = self.subt_manual# bc I don't want to evaluate one conditional repeatedly
            self.kernel1 = np.ones((5,5),np.uint8)
            self.kernel2 = np.ones((17,17),np.uint8)

        else:
            self.subt = self.subt_MOG# bc I don'T want to evaluate one conditional repeatedly

        self._publish = 0
        self.winname = "BGS" # BackgroundSubtractor, not BOGUS
        #self.win2name = "Manual"

# TODO: use tracker
    def add_model(self,model):
        self.model = model
        self.classify = self.model.classify
        self.add_data = self.add_data_md
        self.add_result = self.model.add_result

    def set_frames(self, h=480, w=640):
        self.h, self.w=h, w
        self.out = np.zeros((h,w,3), dtype=np.uint8)
        self.mask = np.zeros((h,w), dtype=np.uint8)

    def init_windows(self):
        cv2.namedWindow(self.winname)
        cv2.moveWindow(self.winname,100,500)
        cv2.createTrackbar('Threshold:',self.winname,self.diff_thresh,255,self.setThreshold)

    def setThreshold(self, val):
        self.diff_thresh=val

    ### Operational logistics
    def hook(self,frame):
        #first_time():
        self.capture_bg(frame)
        self.set_frames(frame.shape[0], frame.shape[1])
        print("Handing off to main process")
        self.hook=self.process
        return False

    def first_time(self,stuff): # PASS STUFF HERE TO SET UP EXTERNALLY):
        ###
        self.hook=self.process

    def loop(self):

        cv2.imshow(self.winname, self.out)
        cv2.imshow("mask", self.mask)

        k = cv2.waitKey(30) & 0xFF
        return self.keyboard_ops(k)

    def shutdown(self):
        if not self.model is None:
            print("Saving data")
            self.model.save_data
        print("Closing UI")
        cv2.destroyAllWindows()
        print("Shutting down")
        sys.exit()
        #if self._drawmanual: #now should be  _drawsecondary
        #    cv2.imshow(self.win2name, self.subt_manual(frame))

    ### Main Process
    def process(self,frame): # design this as this can be just a callback!!
        self.inp=frame
        self.subt(frame)
        self.detect2(self.extract_contours())

    def subt_MOG(self, frame):
        self.mask=self.subtractor.apply(frame)

    def subt_GMG(self, frame):
        fgmask = self.subtractor.apply(frame)
        self.mask=cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

    def subt_watershed(self, frame):
        gray = cv2.cvtColor(cv2.blur(frame,(5,5)),cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray,self.diff_thresh,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.kernel1, iterations = 3)
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,self.kernel2, iterations = 2)

     # sure background area
        #mask = cv2.dilate(mask,self.kernel1,iterations=2)
        self.mask = mask.copy()
     # Finding sure foreground area # We dont need that
     #    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
     #    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
        #sure_fg = np.uint8(sure_fg)
        #self.mask=sure_fg

    def subt_manual(self, frame):

        # Optionla TODO: avg background and decide if img-bg or bg-img
        frame =cv2.blur(frame,(3,3))

        diff = cv2.subtract(self.bg,frame) #self.diff
        (t, mask)=cv2.threshold(cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY), self.diff_thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel2, iterations = 2)

        self.mask=mask
        #reduced number of varaibles, jsut overwriting

    def extract_contours(self):
            self.out = self.inp.copy()

            mask = self.mask
            m2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.N=N=len(contours)
            cv2.putText(self.out, str(N), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,Color.RED,2)

            if cv2.countNonZero(mask) > MIN_PIX:
                m = cv2.moments(mask,1) # we can also get the centroid of the contour
                cx = int(m["m10"] // m["m00"])
                cy = int(m["m01"] // m["m00"])
                cv2.circle(self.out, (cx,cy), 3, Color.BLUE, -1)
                self.results[-2:]=cx,cy

            else:
                self.results[-2:]=-1,-1

            if N==3:
                contours[0]=np.concatenate((contours[0],contours[1],contours[2]))
            if N==2:
                contours[0]=np.concatenate((contours[0],contours[1]))
            if N in {1}:
                return contours[0]
            else:
                return []

    def detect1(self,c):  # I feel like we should get the center in subtract (or right after) and the shape here
        if len(c):
            epsilon = 0.1*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(self.out, [approx],0, Color.MAGENTA, 1)

            if len(c) >= 5:
                ellipse_mask = np.zeros((self.h,self.w), dtype=np.uint8)
                rectangle_mask = ellipse_mask.copy()

                #for mask
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(rectangle_mask,[box], 0, 255, -1)

                ellipse = cv2.fitEllipseAMS(c) #make a fit function iterator
                cv2.ellipse(ellipse_mask ,ellipse , 255, -1)

                rect_err_px=cv2.bitwise_xor(self.mask, rectangle_mask)
                circ_err_px=cv2.bitwise_xor(self.mask, ellipse_mask)

                rect_err=cv2.countNonZero(rect_err_px)
                circ_err=cv2.countNonZero(circ_err_px)

                cv2.putText(self.out, "eR: " +str(rect_err), (400,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,Color.GREEN,2)
                cv2.putText(self.out, "eC: " + str(circ_err), (400,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,Color.YELLOW,2)

                if rect_err < circ_err:
                    #rectangle
                    cv2.drawContours(self.out,[box], 0, Color.GREEN, 2)
                    self.results[0]=0
                    #state="TRACKING"
                    return False

                else:
                    #circle
                    self.results[0]=1
                    cv2.ellipse(self.out,ellipse, Color.YELLOW, 2)
                    #state="TRACKING"
                    return False

            else:
                self.results[0]=-1
                return True
            # (x,y),radius = cv2.minEnclosingCircle(c)
            # cv2.circle(ellipse_mask,(int(x),int(y)),int(radius),(0,255,255),2)
            #deficidnt logic will be here
            #after selection

        else:
            self.results[0]=-1
            return True
        #return False #, (x,y) #1 will be shape

    # try cv2.compare(self.bg,frame,cv2.CMP_LE)

    def detect2(self, c):  # I feel like we should get the center in subtract (or right after) and the shape here

        if len(c):
            rect = cv2.minAreaRect(c)
            (x,y),(w,h), angle = rect
            cv2.putText(self.out, "w: %2.f" % w , (400,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,Color.MAGENTA, 2)
            cv2.putText(self.out, "h: %2.f" % h , (400,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,Color.CYAN, 2)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            if not self._training:
                # can take this inot classify
                res=self.classify(w,h)
                if self.results[0] is not res and res in {0,1}:
                    self.results[0] = res
                    self._publish = 1
                    self.add_result([w,h,res])
                    #later make this apply on init_data
                    #if model is not None: self.model.add_prediction([w,h,res])
            else:
                self.cur_data=[w,h,-5] #current data, with placeholder label


            color=Color.RED if self.results[0] == -1 else Color.GREEN if self.results[0] else Color.YELLOW
            cv2.drawContours(self.out,[box], 0, color, 2)

        else:
            self.results[0]=-1
            return True

    def classify(self,x,y):
        t=0.2
        if x<300 and y<300:
            return (1-t < x/y <1+t)
        else:
            return -1

    def add_result(self,data_row):
        pass #normally we don't store

    def add_data(self,label):
        if self.N in {1,2,3}:
            self.results[0] =label

    def add_data_md(self,label):
        if self.N in {1,2,3}:
            self.results[0] = label
            self.cur_data[2]=label
            self.model.add_data(self.cur_data)
    ### Interface functions

    def capture_bg(self, frame):
        self.bg=frame
        self.means=cv2.mean(frame)
        self.mean=np.mean(self.means)

    def toggle_manual(self):
        self._drawmanual ^= 1
        cv2.destroyWindow(self.win2name) if _drawmanual else cv2.namedWindow(self.win2name)

    # ROS publish flag
    def toggle_training(self):
        self._training ^=1
        print("Training", bool(self._training))


    def print_info(self):
        print("G: ( %.3f, %3.f )" % self.model.Mu, self.model.Sigma )

    def keyboard_ops(self,k):
        if self.external_ops(k): pass
        elif k == ord('c'): self.capture_bg(self.inp)
        elif k == ord('v'): self.toggle_manual()
        #elif k == ord('b'): self.toggle_publish()
        elif k == ord('m'): self.toggle_training()
        elif k == ord('i'): self.print_info()


        elif ord('0') <= k <= ord('1') and self._training: self.add_data(k-ord('0'))
        elif k == ord('2') and self._training: self.add_data(-1)

        elif k == 27 or k == ord('x'): return True
        elif k == ord('p') or k == ord(' '): cv2.waitKey(0)
        return False

    #third party buttons to assign
    def external_ops(self, k):
        return False


def main(source):
    video = cv2.VideoCapture(source)

    # Exit if video not opened.
    if not video.isOpened():
        print "Could not open video"
        sys.exit()
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'Cannot rea1d video file'
        sys.exit()
    else:
        bbm=SimpleBoundingBox(3,frame)
        #bbm.capture_bg(frame)
        #bbm.toggle_manual()
        cv2.namedWindow(bbm.winname)
        cv2.moveWindow(bbm.winname,100,400)
    while (1):
        ok, frame = video.read()
        if ok:
            bbm.hook(frame)
        else: break
        if bbm.loop(): break
    bbm.shutdown()



if __name__ == '__main__':
    print("Running without ROS")
    source=sys.argv[1]
    if source == "0": source=0
    elif source == "1": source=1

    sys.exit(main(source))

else:
    print("Imported Simple Bounding Box module ")
