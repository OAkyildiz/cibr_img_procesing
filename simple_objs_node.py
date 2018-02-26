#!/usr/bin/env python

import sys
from cv_utils.bgs_boundingbox import SimpleBoundingBox
from cv_utils.cv_util_node import CVUtilNode
from cv2 import waitKey


def main(args):

    bbm_agent=SimpleBoundingBox(3)
    bbm_node=CVUtilNode(bbm_agent, "bounding_box_node")
    while(1):
        bbm_node.run()
        waitKey(30)
if __name__ == '__main__':
    sys.exit(main(sys.argv))
