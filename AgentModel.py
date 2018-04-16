import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, \
    InputLayer, Flatten
from keras.optimizers import Adam
from AirSimClient import *

from keras.callbacks import TensorBoard

from collections import deque

from skimage import io

class DQNAgent(object):
    def __init__(self, load_model='./success.model', test_mode=False):
        self.input_shape = (144, 256, 4)
        # why the actions are 7?
        self.nb_actions = 4
        # gamma refers to the discount for future rewards
        self.gamma = 0.9
        # exploration vs. exploitation
        # the fraction of time that dedicate to exploring
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        # tau not clear
        self.tau = .125
        # NOT CLEAR WHAT IS THIS
        # self._explorer = Linear
        self.memory = deque(maxlen=2000)

        self.model = self.create_model()
        self.target_model = self.create_model()

        # print('Try load model weights')
        #if load_model:
        #    self.target_model.load_weights(load_model)
        self.test_mode = test_mode


    def create_model(self):
        model = Sequential()

        model.add(InputLayer(self.input_shape))

        model.add(Conv2D(32, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, 3, padding='same', activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(64, 3, padding='same', activation='relu'))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.nb_actions))

        model.compile(loss='mse',
                      optimizer=Adam())
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if self.test_mode:
            return np.argmax(self.model.predict(np.array([state]))[0])

        if np.random.random() < self.epsilon:
            # env.action_space.sample
            # Returns a array with one sample from each discrete action space
            # designed to perform random actions if less then epsilon
            print('random action')
            return random.randint(0, self.nb_actions - 1)
        print('nn_out action')
        return np.argmax(self.model.predict(np.array([state]))[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        print('----------------replay memory----------------')
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = state[None, :, :, :]
            new_state = new_state[None, :, :, :]

            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = self.target_model.predict(new_state)[0].max()
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        print('----------------model train----------------')
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


class DroneEnv(object):
    def __init__(self):
        """
        init:
        x: 31 - 39,
        y: -4.25 - 7.25
        z: -8
        
        tar:
        x: 34.85
        y: 1.25
        z: -> 0
        
        """
        print('Env Initializing', sep='')
        self.initZ = -8
        self.initX_range = [32.85, 36.85]
        self.initY_range = [-0.75, 3.25]

        self.tarX = 34.85
        self.tarY = 1.25
        self.tarZ = 0

        self.in_ = 1

        self.tolerated_error = 1

        self.speed = 5
        self.scaling_factor = 0.1
        print('...', sep='')
        self.client = MultirotorClient()
        print('...', sep='')
        self.client.confirmConnection()
        print('...', sep='')
        self.client.enableApiControl(True)
        print('...', sep='')
        self.client.armDisarm(True)
        print('...', sep='')
        self.client.takeoff()
        print('Drone takeoff')

        self.reset()

    def reset(self):
        initX = self.initX_range[0] + (self.initX_range[1] - self.initX_range[0]) * random.random()
        initY = self.initY_range[0] + (self.initY_range[1] - self.initY_range[0]) * random.random()
        self.client.moveToPosition(initX, initY,
                                   self.initZ, self.speed)
        
        time.sleep(1)
        print('_____Drone reset_____')

        return self.get_state()

    def get_state(self):
        responses = self.client.simGetImages(
            [
                ImageRequest(3, AirSimImageType.Scene, False, False)
            ]
        )

        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgba = img1d.reshape(response.height, response.width, 4)
        img_rgba = np.flipud(img_rgba)
        io.imsave('./debug.jpg', img_rgba[:, :, :3])
        return img_rgba

    def action_understand(self, action):
        #if action == 0:
        #    quad_offset = (0, 0, 0)
        ep = 0.01
        if action == 0:
            quad_offset = (self.scaling_factor, 0, ep)
        elif action == 1:
            quad_offset = (0, self.scaling_factor, ep)
        #elif action == 3:
        #    quad_offset = (0, 0, self.scaling_factor)
        elif action == 2:
            quad_offset = (-self.scaling_factor, 0, ep)
        elif action == 3:
            quad_offset = (0, -self.scaling_factor, ep)
        #elif action == 6:
        #    quad_offset = (0, 0, -self.scaling_factor)
        return quad_offset

    def step(self, action):
        quad_offset = self.action_understand(action)
        quad_vel = self.client.getVelocity()
        # self.client.moveByVelocity(quad_vel.x_val + quad_offset[0], quad_vel.y_val + quad_offset[1],
        #                            quad_vel.z_val + quad_offset[2], 5)
        self.client.moveByVelocity(quad_offset[0], quad_offset[1],
                                   quad_offset[2], 5)
        # sleep to avoid turbulence
        time.sleep(0.5)
        new_state = self.get_state()
        reward = self.compute_reward()
        done = self.is_done()
        if done:
            reward += 100
        return new_state, reward, done

    def in_bound(self, loc):
        x = loc.x_val
        y = loc.y_val
        z = loc.z_val
        print(x, y, z)
        return (((y - 1.25) ** 2 + (x - 34.85) ** 2) ** 0.5 < -z*4/8) | (z > -1)


    def compute_reward(self):
        loc = self.client.getPosition()
        # print(loc)
        # loss = 1 / ((loc.x_val - self.tarX) ** 2 +
        #      (loc.y_val - self.tarY) ** 2 +
        #     (loc.z_val - self.tarZ) ** 2) ** 0.5
        #x: 31 - 39,
        #y: -4.25 - 7.25
        #z: -8
        # if (loc.x_val < 31)|(loc.x_val > 39)|(loc.y_val < -4.25)|(loc.y_val > 7.25):
        #     loss -= 10

        loss =  - 2 * abs(loc.x_val - self.tarX) - 2 * abs(loc.y_val - self.tarY)
        # (8 + loc.z_val) * 0.1
        self.in_ = self.in_bound(loc)
        if not self.in_:
            loss -= 100
        print("reward:{}".format(loss))
        return loss

    def is_done(self):
        loc = self.client.getPosition()
        if ((loc.x_val - self.tarX) ** 2 +
            (loc.y_val - self.tarY) ** 2 +
            (loc.z_val - self.tarZ) ** 2) ** 0.5 < self.tolerated_error:
            return 1
        else:
            return 0

    def get_dist(self):
        loc = self.client.getPosition()
        return ((loc.x_val - self.tarX) ** 2 +
                (loc.y_val - self.tarY) ** 2 +
                (loc.z_val - self.tarZ) ** 2)

if __name__ == '__main__':

    # initialize env
    # hyper-para init
    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 50

    # Agent init
    agent = DQNAgent()

    # ENV
    env = DroneEnv()

    action_dict = {
         0: 'forward',
         1: 'left',
         2: 'back',
         3: 'right',
        }

    # training
    for trial in range(trials):
        # the agent is only exposed to the bottom camera view
        # for every series of trials, the agent is randomly initialized
        current_state = env.reset()
        for step in range(trial_len):
            # decide which action to take,
            action = agent.act(current_state)
            print('taking action', action_dict[action])
            new_state, reward, done = env.step(action)

            # remember
            agent.remember(current_state, action, reward, new_state, done)
            # internally iterates default (prediction) model
            agent.replay()
            # iterates target model
            agent.target_train()

            current_state = new_state
            if done:
                print('******Complete at step {}******'.format(step))
                break

            if not env.in_:
                print('***************out of range*****************')
                break

        if (step == trial_len - 1) | (not env.in_):
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            agent.save_model("success.model")
