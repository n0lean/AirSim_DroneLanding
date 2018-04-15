from AirSimClient import *
import pprint

# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoff()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))
# client.moveToPosition(0, 0, -8, 3)
# target pos: (34.86, 1.27, -1, 3)
client.moveToPosition(39, 1.27, -8, 3)
client.moveToPosition(39, 7.27, -8, 3)
client.hover()
time.sleep(2)

responses = client.simGetImages(
    [
        ImageRequest(3, AirSimImageType.Scene, False, False)
    ]
)

response = responses[0]
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
img_rgba = img1d.reshape(response.height, response.width, 4)
img_rgba = np.flipud(img_rgba)
client.write_png(os.path.normpath('test.png'), img_rgba)
