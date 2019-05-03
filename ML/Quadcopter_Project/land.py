import numpy as np
from physics_sim import PhysicsSim

class Land():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim([0.0, 0.0, 10.0,0.0,0.0,0.0], init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        if target_pos is None :
            print("Setting default init pose")
        self.target_pos = 0.0
        
        self.max_duration = 5.0 #sec

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        
        if abs(self.target_pos - self.sim.pose[2])< 0.003:
            reward = reward = -abs(self.target_pos - self.sim.pose[2])/100.0
            
        if abs(self.target_pos - self.sim.pose[2])> 5:
            reward =-abs(self.target_pos - self.sim.pose[2])/5.0
        else:
             reward = -abs(self.target_pos - self.sim.pose[2])/10.0
        
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            if done :
                if (self.sim.pose[2] - self.target_pos)<0.5:  # agent has crossed the target height
                    reward += 7.0  # bonus reward
                if (self.sim.pose[2] - self.target_pos)>3:  # agent has crossed the target height
                    reward -= 1.0  # bonus reward
                
                if self.sim.time > self.max_duration:  # agent has run out of time
                    reward -= 10.0  # extra penalty
                if self.sim.time < self.max_duration:  # agent has run out of time
                    reward += 3.0  # bonus reward
                    
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state