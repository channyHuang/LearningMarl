import numpy as np
import pygame

import utils
from Lidar import Lidar

class Obstacle:
    def __init__(self, boundary =2, speed = 0):
        self.position = np.random.uniform(low=0.45, high=boundary-0.55, size=(3, ))
        self.position[-1] = 0 # firstly let z = 0
        self.radius = np.random.uniform(0.1, 0.15)
        self.height = np.random.uniform(0.1, 0.15)

        self.speed = speed
        angle = np.random.uniform(0, 2 * np.pi)
        self.velocity = np.array([self.speed * np.cos(angle), self.speed * np.sin(angle)])
        
    def step(self):
        pass

    def _return_obs_info(self):
        x, y, z = self.position
        r = self.radius
        h = self.height
        return (x,y,z,r,h)

class MultiTarEnv:
    def __init__(self,length,num_obstacle,num_hunters,num_targets,
                 h_actor_dim,t_actor_dim,action_dim,visualize_lasers=False):
        self.boundary = length
        self.length = length # length of boundary

        self.num_obstacle = num_obstacle # number of obstacles
        self.num_hunters = num_hunters # number of hunters
        self.num_targets = num_targets # number of targets

        self.h_actor_dim = h_actor_dim # each hunter's observation dimension
        self.t_actor_dim = t_actor_dim # each target's observation dimension
        self.action_dim = action_dim # dimension of each agent's action

        self.time_step = 0.5 # update time step
        self.v_max = 0.05
        self.a_max = 0.01
        self.num_lasers = 16 # beams of lasers
        self.L_sensor = 0.2 # max length of sensor
        self.escape_distance = 0.05 # escape distance threshold for target
        self.distance_threshold = 0.01 # distance threshold for collision
        self.max_escape_angle = 30 # max escape angle for target (degree)

        ## Instancing hunters and targets, obstacles.
        # set of obstacles
        self.obstacles = [Obstacle() for _ in range(self.num_obstacle)]
        # set of hunters
        self.hunters = [Hunter(self.length, self.L_sensor,self.num_lasers, self.time_step, self.obstacles) for _ in range(self.num_hunters)]
        # set of targets
        self.targets = [Target(self.length, self.L_sensor,self.num_lasers, self.time_step, self.obstacles) for _ in range(self.num_targets)]

        # Control whether to visualize laser scans & plot relevant
        self.visualize_lasers = visualize_lasers

        # reward relevant
        self.capture_reward = 2.0        # roundup success reward
        self.chase_reward_coeff = 0.8    # chase reward coeff
        self.escape_reward_coeff = 0.4   # escape reward coeff
        self.safe_penalty_coeff = 0.7    # safe penalty coeff
        self.direction_reward_coeff = 0.5

        self.screen = None
        self.screen_size = 600
        self.scale = self.screen_size / self.boundary 
        self.steps = 0
        self.icon_agent = None
        self.history_positions = [[] for _ in range(self.num_hunters + self.num_targets)]
    
    def _collect_obs_info(self):
        multi_obs_info = []
        for obstacle in self.obstacles:
            multi_obs_info.append(obstacle._return_obs_info())

    def reset(self):
        for hunter in self.hunters:
            hunter.position = np.random.uniform(low=0.25, high=1.0, size=(3,))  # TODO!: initial spawn scope
            hunter.position[-1] = 0.10  # initial height
            hunter.velocity = np.zeros(3)
            hunter.history_pos = []
            hunter.lasers = hunter.lidar.scan(hunter.position, self.length)

        for target in self.targets:
            target.position = np.random.uniform(low=1.50, high=1.75, size=(3,))  # TODO!: initial spawn scope
            target.position[-1] = 0.10  # initial height
            target.velocity = np.zeros(3)
            target.history_pos = []
            target.lasers = target.lidar.scan(target.position, self.length)

        # Assign initial targets to hunters
        self._assign_targets_to_hunters()

        # Get initial observations
        h_obs, t_obs = self._get_observations()

        return h_obs,t_obs

    def step(self,actions):
        # Apply actions to hunters and targets
        for i, hunter in enumerate(self.hunters):
            hunter.move(actions[i], self.v_max)
        for i, target in enumerate(self.targets):
            target.move(actions[self.num_hunters + i], self.v_max) # Assume "action" set is combination of hunetrs' & targets'

        # Update lasers after movement
        for hunter in self.hunters:
            hunter.lasers = hunter.lidar.scan(hunter.position, self.length)
        for target in self.targets:
            target.lasers = target.lidar.scan(target.position, self.length)

        # Optionally: Boundary blocking (when train agents, 
        # to allow agents traverse the boundary may get worse performance)
        for agent in self.hunters + self.targets:
            agent.position[:2] = np.clip(agent.position[:2], 0, self.length)

        # # Assign targets to hunters
        # self._assign_targets_to_hunters()

        # Compute rewards and check for captures
        rewards, dones = self._compute_rewards()

        # Get next observations
        h_obs_next, t_obs_next = self._get_observations()

        return h_obs_next, t_obs_next, rewards, dones

    # TODO: introduce a density-based assignment method
    def _assign_targets_to_hunters(self):
        num_hunters = self.num_hunters
        num_targets = self.num_targets
        targets = self.targets
        
        hunter_target_distances = []
        for hunter in self.hunters:
            distances = [np.linalg.norm(hunter.position[:2] - target.position[:2]) for target in targets]
            hunter_target_distances.append(distances)
        
        base_hunters_per_target = num_hunters // num_targets
        extra_hunters = num_hunters % num_targets
        hunters_per_target = [base_hunters_per_target + 1 if i < extra_hunters else base_hunters_per_target for i in range(num_targets)]
        
        hunter_indices = list(range(num_hunters))

        for target_idx in range(num_targets):

            num_to_assign = hunters_per_target[target_idx]
            sorted_hunters = sorted(hunter_indices, key=lambda x: hunter_target_distances[x][target_idx])
            assigned_hunters = sorted_hunters[:num_to_assign]
            for hunter_idx in assigned_hunters:
                self.hunters[hunter_idx].assigned_target = targets[target_idx]
                hunter_indices.remove(hunter_idx)

    def _get_nearest_target(self, hunter):
        """
        Find the nearest target to the given hunter.
        Args:
            hunter (Hunter): The hunter to find the nearest target for.
        Returns:
            Target: The nearest target.
        """
        min_dist = float('inf')
        nearest = None
        for target in self.targets:
            dist = np.linalg.norm(hunter.position[:2] - target.position[:2])  # TODO: 2D distance -> 3D
            if dist < min_dist:
                min_dist = dist
                nearest = target
        return nearest
    
    def _get_observations(self):
        """
        Compute observations for all hunters and targets.
        Returns:
            h_obs (list of np.array): Observations for all hunters.
            t_obs (list of np.array): Observations for all targets.
        """
        h_obs = []
        t_obs = []

        # Precompute hunter positions for nearest hunters
        hunter_positions = np.array([hunter.position for hunter in self.hunters])

        # Compute observations for hunters
        for i, hunter in enumerate(self.hunters):
            # Get other hunters' positions excluding itself
            other_hunters = np.delete(hunter_positions, i, axis=0)
            # Find two nearest hunters
            if len(other_hunters) >= 2:
                distances = np.linalg.norm(other_hunters[:, :2] - hunter.position[:2], axis=1)
                nearest_indices = distances.argsort()[:2]
                nearest_hunters = other_hunters[nearest_indices]
            else:
                # If less than two other hunters, pad with zeros
                nearest_hunters = np.zeros((2, 3))
                if len(other_hunters) == 1:
                    nearest_hunters[0] = other_hunters[0]
                    nearest_hunters[1] = np.zeros(3)
                else:
                    nearest_hunters = np.zeros((2, 3))

            # Get velocity
            velocity = hunter.velocity

            # Get assigned target's position and distance
            if hunter.assigned_target is not None:
                target_pos = hunter.assigned_target.position
                distance_to_target = np.linalg.norm(hunter.position[:2] - target_pos[:2])
            else:
                target_pos = np.zeros(3)
                distance_to_target = 0.0

            # Get laser data
            laser_data = hunter.lasers  # Assuming it's a 1D array of size num_lasers

            # Concatenate all observation components
            obs = np.concatenate([
                nearest_hunters.flatten()/self.length,                   # 2 * 3 = 6
                hunter.position/self.length,                             # 3
                velocity/self.v_max,                                     # 3
                target_pos/self.length,                                  # 3
                np.array([distance_to_target])/(np.sqrt(2)*self.length), # 1
                laser_data/self.L_sensor                                 # num_lasers
            ]).astype(np.float32)

            h_obs.append(obs)

        # Precompute hunter positions for targets' observations
        for target in self.targets:
            # Get target's own position and velocity
            own_pos = target.position
            own_vel = target.velocity

            # Find three nearest hunters
            distances = np.linalg.norm(hunter_positions[:, :2] - own_pos[:2], axis=1)
            nearest_indices = distances.argsort()[:3]
            nearest_hunters = hunter_positions[nearest_indices]
            if len(nearest_hunters) < 3:
                # Pad with zeros if less than 3 hunters
                pad_size = 3 - len(nearest_hunters)
                nearest_hunters = np.vstack([nearest_hunters, np.zeros((pad_size, 3))])

            # Get laser data
            laser_data = target.lasers  # Assuming it's a 1D array of size num_lasers

            # Concatenate all observation components
            obs = np.concatenate([
                own_pos/self.length,                    # 3
                own_vel/self.v_max,                     # 3
                nearest_hunters.flatten()/self.length,  # 3 * 3 = 9
                laser_data/self.L_sensor                # num_lasers
            ]).astype(np.float32)

            t_obs.append(obs)

        return h_obs, t_obs

    def _compute_rewards(self):
        """
        Compute rewards for all agents and check for done conditions.
        Returns:
            rewards (list of float): Rewards for all agents (hunters followed by targets).
            dones (list of bool): Done flags for all agents.
        """
        rewards = [0.0] * (self.num_hunters + self.num_targets)
        dones = [False] * (self.num_hunters + self.num_targets)

        # Map each target to its assigned hunters
        target_hunter_groups = {}
        for target in self.targets:
            target_hunter_groups[target] = [hunter for hunter in self.hunters if hunter.assigned_target == target]

        # Reward for hunters chasing and capturing targets
        for target, hunters in target_hunter_groups.items():
            # calculate chasing reward and ifrounded reward
            for hunter in hunters:
                hunter_dir = hunter.velocity[:2]
                if np.linalg.norm(hunter_dir) == 0:
                    hunter_dir_unit = np.zeros(2)
                else:
                    hunter_dir_unit = hunter_dir / np.linalg.norm(hunter_dir)
                target_dir = target.position[:2] - hunter.position[:2]
                if np.linalg.norm(target_dir) == 0:
                    target_dir_unit = np.zeros(2)
                else:
                    target_dir_unit = target_dir / np.linalg.norm(target_dir)
                chase_reward = np.dot(hunter_dir_unit, target_dir_unit)
                hunter_index = self.hunters.index(hunter)
                rewards[hunter_index] += self.chase_reward_coeff * chase_reward

            multi_hunters_pos = [h.position for h in hunters]
            if utils.isRounded(tuple(target.position[:2]), [tuple(row[:2]) for row in multi_hunters_pos], self.L_sensor, self.max_escape_angle):
                for hunter in hunters:
                    hunter_index = self.hunters.index(hunter)
                    rewards[hunter_index] += self.capture_reward
                target_index = self.targets.index(target)
                dones[self.num_hunters + target_index] = True  # to mark target as done
            # for hunter in hunters:
            #     target_index = self.targets.index(target)
            #     hunter_index = self.hunters.index(hunter)
            #     dist = np.linalg.norm(target.position[:2] - hunter.position[:2])
            #     if dist < self.escape_distance:
            #         hunter_index = self.hunters.index(hunter)
            #         rewards[hunter_index] += self.capture_reward
            #         target_index = self.targets.index(target)
            #         dones[self.num_hunters + target_index] = True

        # Reward for targets
        for target in self.targets:
            target_index = self.targets.index(target)
            if dones[self.num_hunters + target_index]:
                # rewards[self.num_hunters + target_index] += 0  # No additional reward if captured
                continue

            hunters = target_hunter_groups[target]
            if len(hunters) != 0:
                distances = [np.linalg.norm(target.position[:2] - hunter.position[:2]) for hunter in hunters]
                nearest_distance = min(distances)
            else:
                nearest_distance = np.inf

            if nearest_distance > self.escape_distance:
                escape_reward = 0.1
            else:
                escape_reward = -0.1
            rewards[self.num_hunters + target_index] += self.escape_reward_coeff * escape_reward
            # escape_reward = 0.1 * (nearest_distance - self.escape_distance)
            # rewards[self.num_hunters + target_index] += self.escape_reward_coeff * escape_reward

            # always lead to boundary corner
            # if len(hunters) >= 1:
            #     hunter_center = np.mean([h.position[:2] for h in self.hunters])
            #     escape_dir = target.velocity[:2]
            #     optimal_dir = target.position[:2] - hunter_center
            #     if np.linalg.norm(escape_dir) > 0 and np.linalg.norm(optimal_dir) > 0:
            #         escape_dir_unit = escape_dir / np.linalg.norm(escape_dir)
            #         optimal_dir_unit = optimal_dir / np.linalg.norm(optimal_dir)
            #         direction_reward = np.dot(escape_dir_unit, optimal_dir_unit)
            #         rewards[self.num_hunters + target_index] += self.direction_reward_coeff * direction_reward


        # reward for safety (u-u)
        # between hunters
        for i in range(self.num_hunters):
            for j in range(i+1, self.num_hunters):
                distance = np.linalg.norm(self.hunters[i].position[:2] - self.hunters[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[i] -= penalty
                    rewards[j] -= penalty

        # between hunters and targets
        for hunter in self.hunters:
            for target in self.targets:
                distance = np.linalg.norm(hunter.position[:2] - target.position[:2])
                if distance < self.distance_threshold:
                    rewards[self.hunters.index(hunter)] -= self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunters + self.targets.index(target)] -= self.safe_penalty_coeff * (self.distance_threshold - distance)
        
        # between targets
        for i in range(self.num_targets):
            for j in range(i+1, self.num_targets):
                distance = np.linalg.norm(self.targets[i].position[:2] - self.targets[j].position[:2])
                if distance < self.distance_threshold:
                    penalty = self.safe_penalty_coeff * (self.distance_threshold - distance)
                    rewards[self.num_hunters + i] -= penalty
                    rewards[self.num_hunters + j] -= penalty
        
        # reward for safety (uav-obstacles)
        for agent in self.hunters + self.targets:
            min_laser_length = min(agent.lasers)
            collision_penalty = -self.safe_penalty_coeff * (self.L_sensor - min_laser_length) / self.L_sensor
            if agent in self.hunters:
                agent_index = self.hunters.index(agent)
            else:
                agent_index = self.targets.index(agent) + self.num_hunters
            rewards[agent_index] += collision_penalty

        return rewards, dones
    
    def rewardNorm(self, rewards):
        """
        normalize rewards for hunters and targets
        """
        h_rewards = rewards[:self.num_hunters]
        t_rewards = rewards[self.num_hunters:]
        
        h_mean = np.mean(h_rewards)
        h_std = np.std(h_rewards)
        if h_std == 0:
            h_normalized = h_rewards
        else:
            h_normalized = (h_rewards - h_mean) / h_std
        
        t_mean = np.mean(t_rewards)
        t_std = np.std(t_rewards)
        if t_std == 0:
            t_normalized = t_rewards
        else:
            t_normalized = (t_rewards - t_mean) / t_std
        
        normalized_rewards = list(h_normalized) + list(t_normalized)
        return normalized_rewards

    def render(self):
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

        for obstacle in self.obstacles:
            cx, cy, cz, radius, h = obstacle._return_obs_info()
            pygame.draw.circle(self.screen, (128, 128, 128), self._scale_position([cx, cy]), radius * (self.screen_size / self.boundary))

        # Draw hunters
        for hunter in self.hunters:
            x, y, z = hunter.position
            self.screen.blit(self.icon_agent, self._scale_position([x, y]))

        # Draw targets
        for target in self.targets:
            x, y, z = target.position
            pygame.draw.circle(self.screen, (255, 0, 0), self._scale_position([x, y]), 8)

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

    def _create_cylinders(self, ax, x, y, z, r, h):
        """
        Create a 3D cylinder to represent obstacles in the environment.
        """
        # Create obstacle as a cylinder in 3D
        # reduce the number of sampling points in the circumferential and height directions 
        # to improve rendering performance
        theta = np.linspace(0, 2 * np.pi, 16) 
        z_vals = np.linspace(z, z + h, 8)   
        theta, z_vals = np.meshgrid(theta, z_vals)
        x_vals = x + r * np.cos(theta)
        y_vals = y + r * np.sin(theta)

        ax.plot_surface(x_vals, y_vals, z_vals, color='black', alpha=0.5)



class AgentBase:
    def __init__(self, boundary_length, max_distance=0.2, num_rays=16, time_step=0.5, obstacles=None):
        self.boundary_length = boundary_length
        self.position = np.random.uniform(low=1.25, high=1.5, size=(3,))  # TODO: initial spawn scope
        self.position[-1] = 0.10  # initial height
        self.velocity = np.zeros(3)  # initial velocity
        self.time_step = time_step  # update time step

        self.lidar = Lidar(max_distance, num_rays, obstacles)
        self.lasers = self.lidar.scan(self.position, self.boundary_length)
        self.lidar.distances = self.lasers
        self.history_pos = []  # to store trajectory

    # update state
    def move(self, action, v_max=0.1):
        ax, ay = action
        if self.lidar.isInObs:
            self.velocity = np.zeros(3)
        else:
            self.velocity[0] += self.time_step * ax
            self.velocity[1] += self.time_step * ay
            velocity_norm = (self.velocity[0]**2 + self.velocity[1]**2)**0.5
            if velocity_norm > v_max:
                scale_factor = v_max / velocity_norm
                self.velocity[0] *= scale_factor
                self.velocity[1] *= scale_factor

        self.position += self.time_step * self.velocity 
        self.lasers = self.lidar.scan(self.position, self.boundary_length)  
        self.history_pos.append(self.position.copy())

class Hunter(AgentBase):
    def __init__(self, boundary_length, max_distance=0.2, num_rays=16, time_step=0.5, obstacles=None):
        super().__init__(boundary_length, max_distance, num_rays, time_step, obstacles)  # inherit base class
        self.role = {'0':'chaser', '1':'predator'}
        '''
            # TODO: define hunter's role, chaser/predator, chaser will chase target's current position, 
                    while predator will use KF (or other way) to predict target's future position and head to it.
                    so difference between these two roles lies in input.
        '''

class Target(AgentBase):
    def __init__(self, boundary_length, max_distance=0.2, num_rays=16, time_step=0.5, obstacles=None):
        super().__init__(boundary_length, max_distance, num_rays, time_step, obstacles)  # inherit base class