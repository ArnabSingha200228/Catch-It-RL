import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)
'''
state definition
(
    self.rect_x / self.ww,
    self.rect_y / self.wh,
    self.item_x / self.ww,
    self.item_y / self.wh,
    dx,
    dy
)
'''
class CatchItEnvOneStep(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(self, render_mode = None):
        super().__init__()
        self.ww, self.wh = 400, 400
        
        self.rect_x, self.rect_y, self.rect_size = 300, 200, 20
        self.rect_speed = 10
        self.item_size = 10

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
       
        self.action_space = spaces.Discrete(4) 

        self.render_mode = render_mode
        self.state = None
        self.item_x = np.random.randint(0, self.ww - self.item_size)
        self.item_y = np.random.randint(50, self.wh - self.item_size)
        self.old_rect_x = None
        self.old_rect_y = None
        self.score = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
    
        # Ensure grid-aligned rectangle position
        self.rect_x = np.random.randint(0, (self.ww - self.rect_size) // self.rect_speed) * self.rect_speed
        self.rect_y = np.random.randint(50 // self.rect_speed, (self.wh - self.rect_size) // self.rect_speed) * self.rect_speed
    
        self.old_rect_x = self.rect_x
        self.old_rect_y = self.rect_y
    
        # Grid-aligned item position
        x_max = (self.ww - self.item_size) // self.rect_speed
        y_min = 50 // self.rect_speed
        y_max = (self.wh - self.item_size) // self.rect_speed
    
        self.item_x = np.random.randint(0, x_max) * self.rect_speed
        self.item_y = np.random.randint(y_min, y_max) * self.rect_speed

        dx = (self.item_x - self.rect_x) / self.ww
        dy = (self.item_y - self.rect_y) / self.wh

        self.state = (
            self.rect_x / self.ww,
            self.rect_y / self.wh,
            self.item_x / self.ww,
            self.item_y / self.wh,
            dx,
            dy
        )
    
        return np.array(self.state, dtype=np.float32), {}
    
    def step(self, action):
        self.steps += 1
        caught = False
        old_dist = np.linalg.norm([self.rect_x - self.item_x, self.rect_y - self.item_y])
    
        self.old_rect_x = self.rect_x
        self.old_rect_y = self.rect_y
    
        # Movement
        if action == 0 and self.rect_x > 0:
            self.rect_x -= self.rect_speed
        if action == 2 and self.rect_x + self.rect_size < self.ww:
            self.rect_x += self.rect_speed
        if action == 3 and self.rect_y > 40:
            self.rect_y -= self.rect_speed
        if action == 1 and self.rect_y + self.rect_size < self.wh:
            self.rect_y += self.rect_speed
    
        new_dist = np.linalg.norm([self.rect_x - self.item_x, self.rect_y - self.item_y])
    
        # catch condition (partial overlap)
        overlap = (
            abs(self.rect_x - self.item_x) < self.item_size and
            abs(self.rect_y - self.item_y) < self.item_size
        )
        
        # Define rectangles
        # agent_rect = pygame.Rect(self.rect_x, self.rect_y, self.rect_size, self.rect_size)
        # item_rect = pygame.Rect(self.item_x, self.item_y, self.item_size, self.item_size)
        # overlap = agent_rect.colliderect(item_rect)
    
        if overlap:
            reward = 4.0
            self.score += 1
            caught = True
            # Respawn item at grid-aligned position
            x_max = (self.ww - self.item_size) // self.rect_speed
            y_min = 50 // self.rect_speed
            y_max = (self.wh - self.item_size) // self.rect_speed
    
            self.item_x = np.random.randint(0, x_max) * self.rect_speed
            self.item_y = np.random.randint(y_min, y_max) * self.rect_speed
        else:
            # Distance reward shaping
            max_possible_dist = np.linalg.norm([self.ww, self.wh])
            norm_dist = new_dist / max_possible_dist
            distance_reward = -norm_dist
    
            # Movement reward
            if new_dist < old_dist:
                movement_reward = 0.3
            else:
                movement_reward = -0.3
    
            # Alignment reward (helps centering)
            centered_x = (self.item_x + self.item_size // 2) >= self.rect_x and (self.item_x + self.item_size // 2) <= (self.rect_x + self.rect_size)
            centered_y = (self.item_y + self.item_size // 2) >= self.rect_y and (self.item_y + self.item_size // 2) <= (self.rect_y + self.rect_size)
            alignment_reward = 0.2 if (centered_x and centered_y) else 0.0
    
            # Combine
            reward = 0.7 * distance_reward + 0.3 * movement_reward + alignment_reward - 0.01  # step penalty

            # oscillation penalty
            if (self.rect_x == self.old_rect_x and self.rect_y == self.old_rect_y):
                oscillation_penalty = -1.0  # agent didn’t move
            elif (action == 0 and self.old_rect_x < self.rect_x) or (action == 2 and self.old_rect_x > self.rect_x) or \
                 (action == 1 and self.old_rect_y > self.rect_y) or (action == 3 and self.old_rect_y < self.rect_y):
                oscillation_penalty = -1.0  # went in reverse
            else:
                oscillation_penalty = 0.0
            
            # Then in final reward
            reward += oscillation_penalty
    
        # Wall penalty
        if (
            (self.rect_x <= 0 or self.rect_x + self.rect_size >= self.ww) or
            (self.rect_y <= 40 or self.rect_y + self.rect_size >= self.wh)
        ):
            reward -= 0.1
    
        # Update state
        dx = (self.item_x - self.rect_x) / self.ww
        dy = (self.item_y - self.rect_y) / self.wh

        self.state = (
            self.rect_x / self.ww,
            self.rect_y / self.wh,
            self.item_x / self.ww,
            self.item_y / self.wh,
            dx,
            dy
        )
        
        done = caught or (self.steps >= 4000)
        
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.state is None:
            return

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.ww, self.wh))
            pygame.display.set_caption("Catch It!")
            self.clock = pygame.time.Clock()
            self.isopen = True
            self.trail_surface = pygame.Surface((self.ww, self.wh))
            self.trail_surface.set_alpha(30)
            self.trail_surface.fill(black)
            self.screen.fill(black)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        font = pygame.font.Font("freesansbold.ttf", 25)
        r = np.random.randint(50, 255)
        g = np.random.randint(50, 255)
        b = np.random.randint(50, 255)
        pygame.draw.rect(self.screen, (r, g, b), (self.old_rect_x, self.old_rect_y, self.rect_size, self.rect_size))
        pygame.time.Clock().tick(120)

        txt = font.render(f"Score : {self.score}", True, (255, 255, 255))
   
        self.screen.blit(self.trail_surface, (0, 0))
        self.screen.blit(txt, (10, 10))
        pygame.draw.rect(self.screen, (255, 255, 0), (self.rect_x, self.rect_y, self.rect_size, self.rect_size))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.item_x, self.item_y, self.item_size, self.item_size))
        pygame.draw.line(self.screen, (255, 255, 255), (0, 35), (self.ww, 35))
        pygame.display.update()

    def close(self):
        if hasattr(self, "isopen") and self.isopen:
            pygame.quit()
            self.isopen = False
        


class CatchItEnvRender(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 120}

    def __init__(self, render_mode = None, timer = False, max_steps = 100):
        super().__init__()
        self.ww, self.wh = 400, 400
        
        self.rect_x, self.rect_y, self.rect_size = 300, 200, 20
        self.rect_speed = 10
        self.item_size = 10
        self.max_steps = max_steps

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
              
        self.action_space = spaces.Discrete(4) 

        self.render_mode = render_mode
        self.state = None
        self.item_x = np.random.randint(0, self.ww - self.item_size)
        self.item_y = np.random.randint(50, self.wh - self.item_size)
        self.old_rect_x = None
        self.old_rect_y = None
        self.start_time = None
        self.score = 0
        self.steps = 0
        self.timer = timer

    def reset(self, seed=None, options=None):
        self.steps = 0
        super().reset(seed=seed)
    
        if self.timer:
            self.start_time = time.time()
    
        self.score = 0
        self.rect_x = np.random.randint(0, self.ww - self.rect_size)
        self.rect_y = np.random.randint(50, self.wh - self.rect_size)
    
        self.old_rect_x = self.rect_x
        self.old_rect_y = self.rect_y
    
        x_max = (self.ww - self.item_size) // self.rect_speed
        y_min = 50 // self.rect_speed
        y_max = (self.wh - self.item_size) // self.rect_speed
    
        self.item_x = np.random.randint(0, x_max) * self.rect_speed
        self.item_y = np.random.randint(y_min, y_max) * self.rect_speed
    
        dx = (self.item_x - self.rect_x) / self.ww
        dy = (self.item_y - self.rect_y) / self.wh

        self.state = (
            self.rect_x / self.ww,
            self.rect_y / self.wh,
            self.item_x / self.ww,
            self.item_y / self.wh,
            dx,
            dy
        )
    
        return np.array(self.state, dtype=np.float32), {}
    
    def step(self, action):
        self.steps += 1
        old_dist = np.linalg.norm([self.rect_x - self.item_x, self.rect_y - self.item_y])
    
        self.old_rect_x = self.rect_x
        self.old_rect_y = self.rect_y
    
        # Movement
        if action == 0 and self.rect_x > 0:
            self.rect_x -= self.rect_speed
        if action == 2 and self.rect_x + self.rect_size < self.ww:
            self.rect_x += self.rect_speed
        if action == 3 and self.rect_y > 40:
            self.rect_y -= self.rect_speed
        if action == 1 and self.rect_y + self.rect_size < self.wh:
            self.rect_y += self.rect_speed
    
        new_dist = np.linalg.norm([self.rect_x - self.item_x, self.rect_y - self.item_y])
    
        # Soft overlap condition (forgiveness)
        overlap = (
            abs(self.rect_x - self.item_x) < self.item_size and
            abs(self.rect_y - self.item_y) < self.item_size
        )
        # # Define rectangles
        # agent_rect = pygame.Rect(self.rect_x, self.rect_y, self.rect_size, self.rect_size)
        # item_rect = pygame.Rect(self.item_x, self.item_y, self.item_size, self.item_size)
        # overlap = agent_rect.colliderect(item_rect)
    
        if overlap:
            reward = 25.0
            self.score += 1
    
            # Respawn item
            x_max = (self.ww - self.item_size) // self.rect_speed
            y_min = 50 // self.rect_speed
            y_max = (self.wh - self.item_size) // self.rect_speed
    
            self.item_x = np.random.randint(0, x_max) * self.rect_speed
            self.item_y = np.random.randint(y_min, y_max) * self.rect_speed
    
        else:
            max_possible_dist = np.linalg.norm([self.ww, self.wh])
            norm_dist = new_dist / max_possible_dist
            distance_reward = -norm_dist
    
            # Directional reward
            movement_reward = 0.1 if new_dist < old_dist else -0.1
    
            # Proximity reward (for minor misalignment)
            proximity_reward = 0.2 if new_dist <= self.rect_speed else 0.0
    
            reward = distance_reward + movement_reward + proximity_reward - 0.01  # time penalty
    
        # Wall penalty
        if (
            (self.rect_x <= 0 or self.rect_x + self.rect_size >= self.ww) or
            (self.rect_y <= 40 or self.rect_y + self.rect_size >= self.wh)
        ):
            reward -= 0.1
    
        dx = (self.item_x - self.rect_x) / self.ww
        dy = (self.item_y - self.rect_y) / self.wh

        self.state = (
            self.rect_x / self.ww,
            self.rect_y / self.wh,
            self.item_x / self.ww,
            self.item_y / self.wh,
            dx,
            dy
        )
    
        done = (self.steps >= self.max_steps) if not self.timer else (time.time() - self.start_time) >= 10
    
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if self.state is None:
            return

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((self.ww, self.wh))
            pygame.display.set_caption("Catch It!")
            self.clock = pygame.time.Clock()
            self.isopen = True
            self.trail_surface = pygame.Surface((self.ww, self.wh))
            self.trail_surface.set_alpha(30)
            self.trail_surface.fill(black)
            self.screen.fill(black)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        font = pygame.font.Font("freesansbold.ttf", 25)
        r = np.random.randint(50, 255)
        g = np.random.randint(50, 255)
        b = np.random.randint(50, 255)
        pygame.draw.rect(self.screen, (r, g, b), (self.old_rect_x, self.old_rect_y, self.rect_size, self.rect_size))
        pygame.time.Clock().tick(120)

        txt = font.render(f"Score : {self.score}", True, (255, 255, 255))
        if self.timer:
            time_str = f"Time : {10 - (round(time.time() - self.start_time))}"
        else:
            time_str = ""
            
        txt_time = font.render(time_str, True, (255, 255, 255))
    
        self.screen.blit(self.trail_surface, (0, 0))
        self.screen.blit(txt, (10, 10))
        self.screen.blit(txt_time, (250, 10))
        pygame.draw.rect(self.screen, (255, 255, 0), (self.rect_x, self.rect_y, self.rect_size, self.rect_size))
        pygame.draw.rect(self.screen, (255, 255, 255), (self.item_x, self.item_y, self.item_size, self.item_size))
        pygame.draw.line(self.screen, (255, 255, 255), (0, 35), (self.ww, 35))
        pygame.display.update()

    def close(self):
        if hasattr(self, "isopen") and self.isopen:
            pygame.quit()
            self.isopen = False
        
