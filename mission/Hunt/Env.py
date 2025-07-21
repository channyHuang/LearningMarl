import numpy as np
import itertools
from gymnasium import spaces
import copy
import time
import pygame

from math_tool import *

class Obstacle():
    def __init__(self, boundary = 2):
        self.position = np.random.uniform(low = 0.25 * boundary, high = boundary * 0.75, size = (2, ))
        self.radius = np.random.uniform(0.05 * boundary, 0.075 * boundary)

class Env:
    def __init__(self, boundary = 2, 
                num_obstacles = 3, 
                num_agents = 4,
                num_targets = 1,
                laser_length = 0.2,
                num_lasers = 16):
        self.boundary = boundary
        self.num_obstacles = num_obstacles
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.num_lasers = num_lasers
        self.laser_length = laser_length if laser_length is not None else 0.1 * self.boundary
        self.step_length = 0.25 * self.boundary
        self.velocity_max_agent = 0.05 * self.boundary
        self.velocity_max_target = 0.06 * self.boundary
        self.dist_capture = 0.15 * self.boundary  

        self.agents_name = []
        self.action_space = {}
        # agent: num_features 4 + (num_agents - 1) * 2 + self lasers 16
        # target: num_
        self.observation_space = {}
        for i in range(self.num_agents):
            if i < self.num_agents - self.num_targets:
                name = f'agent_{i}'
                self.agents_name.append(name)
                self.action_space[name] = spaces.Box(low = -np.inf, high = np.inf, shape = (2, ))
                self.observation_space[name] = spaces.Box(low = -np.inf, high = np.inf, shape = (26, ))
            else:
                name = f'target_{i - (self.num_agents - self.num_targets)}'
                self.agents_name.append(name)
                self.action_space[name] = spaces.Box(low = -np.inf, high = np.inf, shape = (2, ))
                self.observation_space[name] = spaces.Box(low = -np.inf, high = np.inf, shape = (23, ))
        self.obstacles = [Obstacle() for _ in range(self.num_obstacles)]
        self.laser_distances = [[self.laser_length for _ in range(self.num_lasers)] for _ in range(self.num_agents)] 

        self.screen = None
        self.screen_size = 600
        self.scale = self.screen_size / self.boundary 
        self.steps = 0
        self.icon_agent = None
        self.history_positions = [[] for _ in range(self.num_agents)]

    def reset(self):
        self.steps = 0
        self.history_positions = [[] for _ in range(self.num_agents)]

        np.random.seed((int)(time.time()))
        # [0, self.boundary]
        self.positions = []
        self.velocities = np.zeros((self.num_agents, 2))
        for i in range(self.num_agents):
            if i < self.num_agents - self.num_targets:
                self.positions.append(np.random.uniform(low = 0.05 * self.boundary, high = 0.2 * self.boundary, size = (2, )))
            else:
                self.positions.append(np.random.uniform(low = 0.25 * self.boundary, high = 0.9 * self.boundary, size = (2, )))
        self.calCollide()
        observations = self.getObservations()
        return observations

    def getObservations(self):
        observations = []
        observation = []
        observeTarget = [] 
        for i in range(self.num_agents):
            position = self.positions[i]
            velocity = self.velocities[i]
            # self status
            status = [position[0] / self.boundary, position[1] / self.boundary, velocity[0] / self.velocity_max_agent, velocity[1] / self.velocity_max_agent]
            # teamers status
            status_teamers = []
            status_targets = []
            for j in range(self.num_agents):
                if j != i and j < self.num_agents - self.num_targets:
                    pos = self.positions[j]
                    status_teamers.extend([pos[0] / self.boundary, pos[1] / self.boundary])
                elif j >= self.num_agents - self.num_targets:
                    pos = self.positions[j]

                    dist = np.linalg.norm(position - pos)
                    theta = np.arctan2(position[1] - pos[1], position[0] - pos[0])
                    status_targets.extend([dist / (np.sqrt(2) * self.boundary), theta])
                    if i < self.num_agents - self.num_targets:
                        observeTarget.append(dist / (np.sqrt(2) * self.boundary))
            # dim = self.num_lasers
            lasers = self.agent_lasers[i]
            if i < self.num_agents - self.num_targets:
                observation = [status, status_teamers, lasers, status_targets]
            else:
                observation = [status, lasers, observeTarget]
            observations.append(list(itertools.chain(*observation)))
        return observations

    def step(self, actions):
        self.steps += 1
        dist2target = []
        for i in range(self.num_agents):
            position = self.positions[i]
            if i < self.num_agents - self.num_targets:
                posTarget = self.positions[-1]
                dist2target.append(np.linalg.norm(position - posTarget))

            self.velocities[i] += actions[i] * self.step_length
            velNorm = np.linalg.norm(self.velocities[i])
            if i < self.num_agents - self.num_targets:
                if velNorm > self.velocity_max_agent:
                    self.velocities[i] = self.velocities[i] / velNorm * self.velocity_max_agent
            else:
                if velNorm > self.velocity_max_target:
                    self.velocities[i] = self.velocities[i] / velNorm * self.velocity_max_target
            self.positions[i] += self.velocities[i] * self.step_length
        
        collided = self.calCollide()
        rewards, dones = self.calRewards(collided, dist2target)
        observations = self.getObservations()
        return observations, rewards, dones

    def calRewards(self, IsCollied, last_d):
        dones = [False] * self.num_targets
        rewards = np.zeros(self.num_agents)
        mu1 = 0.7 # r_near
        mu2 = 0.4 # r_safe
        mu3 = 0.01 # r_multi_stage
        ratio_capture = 5 # r_finish
        d_limit = 0.75

        ## 1 reward for single rounding-up-UAVs:
        for i in range(self.num_agents - self.num_targets):
            pos = self.positions[i]
            vel = self.velocities[i]
            velNorm = np.linalg.norm(vel)

            pos_target = self.positions[-1]
            dire_vec = pos_target - pos
            d = np.linalg.norm(dire_vec)

            cos_v_d = np.dot(vel, dire_vec) / (velNorm * d + 1e-3)
            r_near = abs(2 * velNorm / self.velocity_max_agent) * cos_v_d
            # r_near = min(abs(v_i/self.v_max)*1.0/(d + 1e-5),10)/5
            rewards[i] += mu1 * r_near # TODO: if not get nearer then receive negative reward
        
        ## 2 collision reward for all UAVs:
        for i in range(self.num_agents):
            if IsCollied[i]:
                r_safe = -10
            else:
                lasers = self.agent_lasers[i]
                # r_safe = (min(lasers) - self.L_sensor - 0.1)/self.L_sensor
                r_safe = (min(lasers) - self.laser_length - 0.1)/self.laser_length
            rewards[i] += mu2 * r_safe

        ## 3 multi-stage's reward for rounding-up-UAVs
        p0 = self.positions[0]
        p1 = self.positions[1]
        p2 = self.positions[2]
        pe = self.positions[-1]
        S1 = cal_triangle_S(p0,p1,pe)
        S2 = cal_triangle_S(p1,p2,pe)
        S3 = cal_triangle_S(p2,p0,pe)
        S4 = cal_triangle_S(p0,p1,p2)
        d1 = np.linalg.norm(p0-pe)
        d2 = np.linalg.norm(p1-pe)
        d3 = np.linalg.norm(p2-pe)
        Sum_S = S1 + S2 + S3
        Sum_d = d1 + d2 + d3
        Sum_last_d = sum(last_d)

        def isClose(a, b):
            return np.abs(a - b) < 1e-4

        # 3.1 reward for target UAV:
        rewards[-1] += np.clip(10 * (Sum_d - Sum_last_d),-2,2)
        # 3.2 stage-1 track
        if Sum_S > S4 and Sum_d >= d_limit and all(d >= self.dist_capture for d in [d1, d2, d3]):
            r_track = - Sum_d/max([d1,d2,d3])
            rewards[0:2] += mu3*r_track
        # 3.3 stage-2 encircle
        elif Sum_S > S4 and (Sum_d < d_limit or any(d >= self.dist_capture for d in [d1, d2, d3])):
            r_encircle = -1/3*np.log(Sum_S - S4 + 1)
            rewards[0:2] += mu3*r_encircle
        # 3.4 stage-3 capture
        elif Sum_S == S4 and any(d > self.dist_capture for d in [d1,d2,d3]):
            r_capture = np.exp((Sum_last_d - Sum_d)/(3*self.velocity_max_agent))
            rewards[0:2] += mu3*r_capture

        ## 4 finish rewards
        if isClose(Sum_S, S4) and all(d <= self.dist_capture for d in [d1,d2,d3]):
            rewards[0:2] += ratio_capture * 10
            dones = [True] * self.num_agents

        return rewards, dones

    def calCollide(self):
        self.agent_lasers = []
        agent_collideds = [False] * self.num_agents
        for i in range(self.num_agents):
            position = self.positions[i]
            agent_lasers = [self.laser_length] * self.num_lasers
            agent_collided = False
            for obstacle in self.obstacles:
                posObstacle = obstacle.position
                radius = obstacle.radius

                lasers, collided = update_lasers(position, posObstacle, radius, self.laser_length, self.num_lasers, self.boundary)
                agent_collided = collided or agent_collided
                agent_lasers = [min(l, nl) for l, nl in zip(lasers, agent_lasers)]

            if agent_collided:
                self.velocities[i] = np.zeros(2)
                agent_collideds[i] = agent_collided
            self.agent_lasers.append(agent_lasers)    
                    
        return agent_collideds

    def render(self, bSave = False):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('HG title')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 36)

            icon_agent = pygame.image.load("UAV.png").convert_alpha()
            self.icon_agent = pygame.transform.scale(icon_agent, (20, 20))

        self.screen.fill((255, 255, 255))
        border_rect = pygame.Rect(0, 0, self.screen_size, self.screen_size)
        pygame.draw.rect(self.screen, (0, 0, 0), border_rect, 2)

        for obs in self.obstacles:
            pos = obs.position
            radius = obs.radius
            pygame.draw.circle(self.screen, (128, 128, 128), self._scale_position(pos), radius * (self.screen_size / self.boundary))

        for i in range(self.num_agents):
            pos = self.positions[i]
            self.history_positions[i].append( self._scale_position(pos) )
            if i < self.num_agents - self.num_targets:
                self.screen.blit(self.icon_agent, self._scale_position(pos))
            else:
                pygame.draw.circle(self.screen, (255, 0, 0), self._scale_position(pos), 8)

        for i in range(self.num_agents):
            if len(self.history_positions[i]) > 1:
                pygame.draw.lines(self.screen, (0, 0, 255), False, self.history_positions[i] , 2)

        step_text = self.font.render(f"Step: {self.steps}", True, (0, 0, 0))
        self.screen.blit(step_text, (10, 10))

        if bSave:
            pygame.image.save(self.screen, f"frames/{self.steps:04d}.png")

        pygame.display.flip()
        self.clock.tick(30)    

    def _scale_position(self, pos: np.ndarray):
        x = (pos[0] ) * self.scale
        y = (self.boundary - pos[1]) * self.scale
        return (int(x), int(y))

    def close(self)->None:
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.font = None    
