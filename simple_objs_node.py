#!/usr/bin/env python

import rospy, sys
from cv_utils.bgs_boundingbox import SimpleBoundingBox
from cv_utils.cv_util_node import CVUtilNode
from cv_utils.model import Model

#from reflex_msgs.msg import PoseCommand

from cv2 import waitKey



#
# def hand_control(self, k):
#     if k == ord('t'): print("grasp")
#     elif k == ord('y'): print("release")
#     else: return False

def main(args):

    bbm_agent=SimpleBoundingBox(4)
    bbm_agent.add_model(Model(['w','h','label'],"grasp_data"))
    bbm_node=CVUtilNode(bbm_agent, "bounding_box_node")
    #bbm_node.attach_pub('/gripper/command', PoseCommand)
    #bbm_node.attach_controls(hand_control)

    while(1):
        bbm_node.run()

        #waitKey(30)
if __name__ == '__main__':
    sys.exit(main(sys.argv))
