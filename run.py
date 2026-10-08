import argparse
import pygame
from device import asnumpy
from scene import Scene
from render import Renderer
from settings import get_settings


parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, help='path of scene xml file')
parser.add_argument('--config', type=str, default='./conf/settings.xml', help='path of config xml file')
args = parser.parse_args()
scene_xml = args.scene
conf = args.config
settings = get_settings(conf)
if __name__ == '__main__':
    pygame.init()
    scene = Scene.from_xml(scene_xml)
    W, H = scene.camera.W, scene.camera.H
    render = Renderer(scene, settings)
    color = asnumpy(render.render())
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    buffer_surface = pygame.Surface((W, H))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.surfarray.blit_array(buffer_surface, color)
        screen.blit(buffer_surface, (0, 0))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()