import sys
import rospy
#from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#make int msgs

class CVUtilNode: # abstarct this, it can easily work with other cv_utils and be an image bbm_node
    def __init__(self, util, name="cv_util_node", pub_topic=False):
        #self.obj_pub = rospy.Publisher("image_topic_2", ***)
        self.bridge = CvBridge()
        self.util=util
        self.name=name

        rospy.init_node(self.name, anonymous=True)
        self.rate=rospy.Rate(30)
        self.image_sub = rospy.Subscriber("image_topic", Image, self.callback)
        #self.result_pub = rospy.Publisher(self.name+"/results", Floats, queue_size=10) #always publish data
        self.result_msgs = [-1,-1,-1] #make int msgs

        self.pubs=[]
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
        self.util.result_pub.publish(util.results) #try catch


    def img_pub(cv_image): # to handleconverting from OpenCV to ROS
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

    def run(self):
        while not rospy.is_shutdown():
            try:
                
                if self.util.loop(): break
                if not -1 in self.util.results():
                   self.data_pub()
                #self.rate.sleep()
            except KeyboardInterrupt:
                self.util.shutdown()

        self.util.shutdown()

    #adds a publisher to alirlaes,

    def attach_pub(topic, type):
        pubs.append(rospy.Publisher(topic, type))

    def attach_sub(topic, cb_handle):
        subs.append = rospy.Subscriber(topic, type, cb_handle)
