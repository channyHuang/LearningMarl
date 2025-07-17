import numpy as np
import pygame
import threading
from typing import Tuple

class Renderer:
    metadata = {"render_modes": ["human"], "render_fps": 1}
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(Renderer, "_instance"):
            with Renderer._instance_lock:
                if not hasattr(Renderer, "_instance"):
                    Renderer._instance = object.__new__(cls)
        return Renderer._instance

    def __init__(self, render_mode = "human"):
        self.render_mode = render_mode
        self.screen = None
        self.screen_size = 600
        self.scale = 0.5
        self.font = None

        self.num_targets = 3 
        self.num_features = 2
        self.targets = np.random.normal(-1, 1, size = self.num_targets * self.num_features)
        self.targets = self.targets.reshape(self.num_targets, self.num_features)

    def reset(self):
        self.screen = None
        self.screen_size = 600
        self.scale = 0.5

        self.num_targets = 3 
        self.num_features = 2
        self.targets = np.random.normal(-1, 1, size = self.num_targets * self.num_features)
        self.targets = self.targets.reshape(self.num_targets, self.num_features)

    def update(self, x):
        dimx, dimy = x.shape
        self.num_targets = dimx
        self.num_features = dimy

        assert(self.num_features, 2)

        self.targets = x

    def render(self, text = 'Null'):
        if self.render_mode is None:
            return 
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Window Title")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 24)

        self.screen.fill((255, 255, 255))

        # border
        border_rect = pygame.Rect(0, 0, self.screen_size, self.screen_size)
        pygame.draw.rect(self.screen, (0, 0, 0), border_rect, 2)

        # draw targets
        for i, target in enumerate(self.targets):
            # pygame.draw.circle(
            #     self.screen, (0, 0, 255),
            #     self._scale_position(target), 5
            # )
            pygame.draw.circle(self.screen, (0, 0, 255), target, 5)
            id = self.font.render(str(i+1), True, (0, 0, 0))
            self.screen.blit(id, (target[0] - 5, target[1] - 8))

        # show text
        info_text = [
            f"Intent: {text}",
        ]

        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 30))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    # [-1, 1]
    def _scale_position(self, pos: np.ndarray) -> Tuple[int, int]:
        x = (pos[0] * self.screen_size + self.screen_size) * self.scale
        y = (self.screen_size - pos[1] * self.screen_size) * self.scale
        return (int(x), int(y))

if __name__ == '__main__':
    renderer = Renderer()
    while True:
        num_targets = 3
        num_teamers = 5
        targets = np.random.randn(num_targets, 2) * renderer.screen_size
        teamers = np.random.randn(num_teamers, 2) * renderer.screen_size
        for _ in range(20):
            renderer.update(teamers, targets)

            renderer.render()
    