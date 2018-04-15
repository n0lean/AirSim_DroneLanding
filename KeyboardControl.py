from AirSimClient import *
import pprint

# To find the start point of the env.
# target postion is
# 'x_val': 34.859657287597656,
# 'y_val': 1.2473498582839966,
# 'z_val': -0.14795935153961182


OFFSET_MIN = 0.1

client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
# s = pprint.pformat(state)
# print("state: %s" % s)

# AirSimClientBase.wait_key('Press any key to takeoff')
client.takeoff()

# AirSimClientBase.wait_key('Press any key to move vehicle to (-10, 10, -10) at 5 m/s')
client.moveToPosition(36, -1, -8, 3)

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

x_offset = 0.1
y_offset = 0.1
z_offset = 0.1
# client.moveByVelocity(1, 0, 0, 5)

while 1:
    # print(quad_vel)
    # print(k)
    # print(x_offset)
    state = client.getMultirotorState()
    print("state: %s" % pprint.pformat(state))
    if k == b'q':
        break
    elif k == b'w':
        # x_offset += OFFSET_MIN
        client.moveByVelocity(x_offset, 0, 0, 3)

    elif k == b's':
        # x_offset -= OFFSET_MIN
        client.moveByVelocity(-x_offset, 0, 0, 3)

    elif k == b'a':
        # y_offset += OFFSET_MIN
        client.moveByVelocity(0, y_offset, 0, 3)

    elif k == b'd':
        # y_offset -= OFFSET_MIN
        client.moveByVelocity(0, -y_offset, 0, 3)

    elif k == b'u':
        # z_offset += OFFSET_MIN
        client.moveByVelocity(0, 0, z_offset, 3)

    elif k == b'j':
        # z_offset -= OFFSET_MIN
        client.moveByVelocity(0, 0, -z_offset, 3)

    elif k == b'z':
        client.hover()
        k = AirSimClientBase.wait_key()

    # z_offset = - quad_vel.z_val
    time.sleep(0.5)
