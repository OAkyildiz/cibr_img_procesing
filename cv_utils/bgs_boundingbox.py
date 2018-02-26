import sys
import numpy as np
import cv2

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

    BG_METHODS=[cv2.bgsegm.createBackgroundSubtractorMOG,
     cv2.createBackgroundSubtractorMOG2,
     cv2.bgsegm.createBackgroundSubtractorGMG, nothing]
    def __init__(self,method=MAN, bg=[]):
        self.subtractor=SimpleBoundingBox.BG_METHODS[method]()
        self.out=np.zeros((480,640,3), dtype=np.uint8)

        self.results=[0,0,0]

        if method == SimpleBoundingBox.GMG:
            self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
            self.subt = self.subt_GMG

        elif method == SimpleBoundingBox.MAN:
            #if bg: self.capture_bg(bg)
            self.subt = self.subt_manual# bc I don'T want to evaluate one conditional repeatedly
            self.kernel1 = np.ones((5,5),np.uint8)
            self.kernel2 = np.ones((15,15),np.uint8)

        else:
            self.subt = self.subt_MOG# bc I don'T want to evaluate one conditional repeatedly

        self._publish =1
        self.winname = "BGS" # BackgroundSubtractor, not BOGUS
        #self.win2name = "Manual"

        #cv2.namedWindow("BGS")
        #cv2.moveWindow("BGS",100,1200)

        #
# TODO: use tracker?
# that exits on ros_shutdown but waits on vide is not open
    #packaged functionality for ease of use

    # call this ffom outside, it will do the first time setup and then hand it to the process

    ### Operational logistics
    def hook(self,frame):
        #first_time():
        self.capture_bg(frame)
        print("Handing off to main process")
        self.hook=self.process
        return False

    def first_time(self,stuff): # PASS STUFF HERE TO SET UP EXTERNALLY):
        ###
        self.hook=self.process

    def loop(self):
        cv2.imshow(self.winname, self.out)
        k = cv2.waitKey(30) & 0xFF
        return self.keyboard_ops(k)



    def shutdown(self):
        print("Closing UI")
        cv2.destroyAllWindows()
        print("Shutting down")
        sys.exit()
        #if self._drawmanual: #now should be  _drawsecondary
        #    cv2.imshow(self.win2name, self.subt_manual(frame))

    ### Main Process
    def process(self,frame): # design this as this can be just a callback!!
        self.out=frame
        fg=self.subt(frame)

        #N=len(contours)
        #cv2.drawContours(frame, contours,-1, (0,255,0), 4)
        return self.detect(fg)

    def subt_MOG(self, frame):
        return self.subtractor.apply(frame)

    def subt_GMG(self, frame):
        fgmask = self.subtractor.apply(frame)
        return cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)

    def subt_manual(self, frame):
        # Optionla TODO: avg background and decide if img-bg or bg-img
        frame =cv2.blur(frame,(3,3))

        diff = cv2.subtract(self.bg,frame)
        (t, mask)=cv2.threshold(cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY), 53, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel2)

        #reduced number of varaibles, jsut overwriting
        return mask

    def detect(self, mask):  # I feel like we should get the center in subtract (or right after) and the shape here
        m2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        N=len(contours)
        if cv2.countNonZero(mask) > MIN_PIX:
            m = cv2.moments(mask,1) # we can also get the centroid of the contour
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.circle(self.out, (cx,cy), 2, (10,10,10), -1)
            self.results[-2:]=cx,cy

        else:
            self.results[-2:]=-1,-1
            return True

 
		
        cv2.putText(self.out, str(N), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        if N in {1,2}:
            c=contours[0]
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(self.out, (x, y), (x + w, y + h), (255, 0, 0),  3, 1)

            epsilon = 0.1*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(self.out, [approx],0, (255,0,255), 1)

            hull = cv2.convexHull(c)
            cv2.drawContours(self.out, [hull],0, (255,255,0), 1)

            #for mask
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.out,[box],0,(0,255,0),2)

            if len(c) >= 5:
                ellipse = cv2.fitEllipseAMS(c) #make a fit function iterator
                cv2.ellipse(self.out,ellipse,(0,0,255),3)

            (x,y),radius = cv2.minEnclosingCircle(c)
            cv2.circle(self.out,(int(x),int(y)),int(radius),(0,255,255),2)
            #deficidnt logic will be here
            #after selection
            self.results[0]=1
        else:
            self.results[0]=-1
            return True

        return False #, (x,y) #1 will be shape

    # try cv2.compare(self.bg,frame,cv2.CMP_LE)
    #def pub(self):this isros functionality, not wanted here
    #    pass

    ### Interface functions

    def capture_bg(self, frame):
        self.bg=frame
        self.means=cv2.mean(frame)
        self.mean=np.mean(self.means)

    def toggle_manual(self):
        self._drawmanual ^= 1
        cv2.destroyWindow(self.win2name) if _drawmanual else cv2.namedWindow(self.win2name)

    # ROS publish flag
    def toggle_publish(self):
        self._publish ^=1

    def keyboard_ops(self,k):
        if k == ord('c'): self.capture_bg(self.out)
        elif k == ord('v'): self.toggle_manual()
        elif k == ord('b'): self.toggle_publish()
        elif k == 27 or k == ord('x'): return True
        elif k == ord('p') or k == ord(' '): cv2.waitKey(0)
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
        print 'Cannot read video file'
        sys.exit()
    else:
        bbm=SimpleBoundingBox(3,frame)
        #bbm.capture_bg(frame)
        #bbm.toggle_manual()

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
    sys.exit(main(source))

else:
    print("Imported Simple Bounding Box module ")
