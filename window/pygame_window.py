import pygame
from numpy import ndarray
from .window import Window


class PygameWindow(Window):
    def __init__(self, width: int, height: int, title: str):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.surface = pygame.Surface((width, height), depth=24)
        self.running = True

    def update(self, pixels: ndarray):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
        pygame.surfarray.blit_array(self.surface, pixels)
        self.screen.blit(self.surface, (0, 0))
        pygame.display.flip()

    def wait(self):
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()