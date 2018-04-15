from skimage import io
from AgentModel import DQNAgent, DroneEnv
import numpy as np


if __name__ == '__main__':
    save_path = './img_save/'
    npy_path = './npy_save/'
    agent = DQNAgent()
    # DroneEnv.get_dist()
    # need an api for distance info
    # api:
    #   return: float
    env = DroneEnv()
    trial_len = 100
    current_state = env.reset()
    distance_list = []
    for step in range(trial_len):
        action = agent.act(current_state)
        io.imsave(save_path + '{}act{}.jpg'.format(step, action), current_state)
        distance_list.append(env.get_dist())
        new_state, reward, done = env.step(action)
        current_state = new_state
        if done:
            print('Finished in {} steps.'.format(step))
            break
    distance_list = np.array(distance_list)
    np.save(npy_path + '.npy', distance_list)
