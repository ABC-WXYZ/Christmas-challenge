import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

class CliffWalking():
    
    def __init__(self, γ=0.99, alpha=0.1, epsilon=0.1):
        self.alpha = alpha
        self.gamma = γ
        self.epsilon = epsilon
        self.returns = {} # Map (State, Action) to all it's returns in a list
        # each state is a (x, y) 2-tuple
        self.Q = {} # Map (State, Action) to it's average return
        self.FinalPolicy = {} # Store optimal action for each state
        self.FinalStateValues = {} # Store optimal values for each state
        self.env = gym.make('CliffWalking-v1')
        self.rew_sum_list = []
        
    def pi(self, curr_state):
        # epsilon-greedy policy
        action = 0
        if random.random() < self.epsilon:
            action = random.choice([0, 1, 2, 3])
        else:
            # greedy
            q_up = self.Q.get((curr_state, 0), 0)
            q_rt = self.Q.get((curr_state, 1), 0)
            q_dn = self.Q.get((curr_state, 2), 0)
            q_lf = self.Q.get((curr_state, 3), 0)
            action = np.argmax([q_up, q_rt, q_dn, q_lf])

        return action

    def playEpisode(self, verbose):
        curr_state = (self.env.reset())[0]
        curr_action = self.pi(curr_state)
        # print("The starting state is ", curr_state)

        done = False
        rew_sum = 0
        steps = 0
        while (not done) and steps < 1000 : # max_ep_len
            res = self.env.step(curr_action)
            new_state = res[0]
            rew = res[1]
            rew_sum += rew
            done = res[2] or res[3]
            U = 0
            new_action = 0

            if not done:
                # non-terminal state
                new_action = self.pi(new_state)
                U = rew + self.gamma * self.Q.get((new_state, new_action), 0)
            else:
                # terminal state
                U = rew

            self.Q[(curr_state, curr_action)] = self.Q.get((curr_state, curr_action), 0) + self.alpha * (U - self.Q.get((curr_state, curr_action), 0))
            curr_state = new_state
            if not done:
                curr_action = new_action
            steps += 1

        self.rew_sum_list.append(rew_sum)

    def optimize(self):
        for state in range(37):
            q_up = self.Q.get((state, 0), 0)
            q_rt = self.Q.get((state, 1), 0)
            q_dn = self.Q.get((state, 2), 0)
            q_lf = self.Q.get((state, 3), 0)
            q_list = [q_up, q_rt, q_dn, q_lf]
            opt_action = np.argmax(q_list)
            opt_val = max(q_list)
            self.FinalPolicy[state] = opt_action
            self.FinalStateValues[state] = opt_val

    def plot(self):
        # plot opt action
        print("Optimal Actions: ")
        for state in self.FinalPolicy:
            print(self.FinalPolicy[state], end ="")
            if state % 12 == 11 : print() 
        print()
        print("Optimal values: ")
        for state in self.FinalStateValues:
            val = self.FinalStateValues[state]
            formatted_value = f"{val:6.2f}"
            print(formatted_value, end =" ")
            if state % 12 == 11 : print() 



    def play(self, numRounds, verbose=False):
        for i in range(numRounds):
            if i % (numRounds / 10) == 0:
                print(f'Round {i} / {numRounds}')
            self.playEpisode(verbose)
        self.optimize() 
        self.plot()


cw = CliffWalking()
cw.play(10000)
