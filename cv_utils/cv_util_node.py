import sys
import rospy
import types

#from std_msgs.msg import String
from sensor_msgs.msg import Image
from cibr_img_processing.msg import Ints
from cv_bridge import CvBridge, CvBridgeError
#make int msgs
#TODO: get the img size from camera_indo topics
class CVUtilNode: # abstarct this, it can easily work with other cv_utils and be an image bbm_node
    def __init__(self, util, name="cv_util_node", pub_topic=False):
        #self.obj_pub = rospy.Publisher("image_topic_2", ***)
        self.bridge = CvBridge()
        self.util=util
        self.name=name

        rospy.init_node(self.name, anonymous=True)
        self.rate=rospy.Rate(30)
        self.image_sub = rospy.Subscriber("image_topic", Image, self.callback)
        self.result_pub = rospy.Publisher("results", Ints, queue_size=10) #always publish data
        self.result_msgs = [-1,-1,-1] #make int msgs

        self.pubs=lambda:0
        self.subs=[]


        if pub_topic:
            self.image_pub = rospy.Publisher(pub_topic,Image, queue_size=10)

            pass #do stuff with img.pub

    def callback(self,data):
        try:
            self.util.hook(self.bridge.imgmsg_to_cv2(data, "bgr8"))
        except CvBridgeError as e:
            print(e)
    def data_pub(self):
        self.result_pub.publish(self.util.results) #try catch


    def img_pub(cv_image): # to handleconverting from OpenCV to ROS
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def run(self):
        self.util.init_windows()
        while not rospy.is_shutdown():
            try:

                if self.util.loop(): break
                if not -1 in self.util.results and self.util._publish:
                   self.data_pub()
                   self.util._publish = 0
                   # if self.util._publish:
                   #     for pub in self.pubs:
                   #         pub.publish
                #self.rate.sleep()
            except KeyboardInterrupt:
                self.util.shutdown()

        self.util.shutdown()

    #adds a publisher to alirlaes,

    def attach_pub(self, topic, type):
        self.pubs.pub.append(rospy.Publisher(topic, type, queue_size=1))
        # TODO:attach structs of publisher and message template instead
        # so it is iterable together
        #pubs.pub=... pubs.msg=type()
    def attach_sub(self, topic, cb_handle):
        self.subs.append = rospy.Subscriber(topic, type, cb_handle)

    def attach_controls(self, fun_handle):
        # bind the method to instance
        self.util.external_ops=types.MethodType(fun_handle,self.util)
