#! /usr/bin/python

# A mouse handler clase to add to your UIs, you can override methods to add your own functionality


class MouseHandler(object):

    def __init__(self, namedWindow):
        self.windows = namedWindow
        cv2.setMouseCallback(self.windows, self.mouseHandler)


    def mouseHandler(self, event,x,y,flags,param):

        # consider combos (esp. move+button) (oh add state for button downs)
        if event == cv2.EVENT_MOUSEMOVE:
            self.move_method()
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.lbd_method()

        elif event == cv2.EVENT_LBUTTONUP:
            self.lbu_method()

        if event == cv2.EVENT_RBUTTONDOWN:
            self.rbd_method()

        elif event == cv2.EVENT_MOUSEMOVE:
            self.move_method()

        elif event == cv2.EVENT_RBUTTONUP:
            self.rbu_method()

        if event == cv2.EVENT_MBUTTONDOWN:
            self.mbd_method()
    
