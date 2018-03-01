import sys, rospy, cv2
from reflex_msgs.msg import PoseCommand
from cibr_img_processing.msg import Ints

global cmd, is_grasping
is_grasping=False
cmd=PoseCommand()
cmd.f1=0.0
cmd.f2=0.0
cmd.f3=0.0

def vission_cb(msg):
    global cmd, is_grasping
    data=msg.data

    shape=(1.0 if data[0] else 0.0)

    if cmd.preshape == shape:
        pass
    else:
        cmd.preshape=shape
        if not is_grasping: command_pub.publish(cmd)


def handy():
    global cmd, is_grasping

    if not is_grasping:
        cmd.f1=1.99
        cmd.f2=1.99
        cmd.f3=1.99
        command_pub.publish(cmd)
        is_grasping=True
        print("Grasping")
    else:
        cmd.f1=0.0
        cmd.f2=0.0
        cmd.f3=0.0
        command_pub.publish(cmd)
        is_grasping=False
        print("Releasing")

    #rospy.sleep()
def switch_preshape():
    global cmd, is_grasping

    cmd.preshape=(cmd.preshape+1)%3
    print("Manaully switching preshape...")
    if not is_grasping:  command_pub.publish(cmd)


def hand_control(k):
    if k == ord('t'): handy()
    elif k == ord('y'): switch_preshape()
    elif k == ord('x'): return True


    else: return False

def close():
    print("Closing UI")
    cv2.destroyAllWindows()
    print("Shutting down")
    sys.exit()

rospy.init_node("grasp_node",anonymous=True)
command_pub = rospy.Publisher('/left_hand/command_position', PoseCommand, queue_size=1)
vision_sub = rospy.Subscriber('/boxer/results', Ints, vission_cb)

t=10
rate=rospy.Rate(t)
cv2.namedWindow("grasp_control")

while not rospy.is_shutdown():
    try:
        k = cv2.waitKey(t)  & 0xFF
        if hand_control(k):
            close()


    except KeyboardInterrupt:
        close()

    #rate.sleep()
