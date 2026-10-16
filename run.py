import argparse
from window import PygameWindow
from scene import Scene
from render import Renderer
from settings import get_settings


parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, help='path of scene xml file')
parser.add_argument('--config', type=str, default='./conf/settings.xml', help='path of config xml file')
args = parser.parse_args()
scene_xml: str = args.scene
conf = args.config
settings = get_settings(conf)
if __name__ == '__main__':
    scene = Scene.from_xml(scene_xml)
    W, H = scene.camera.W, scene.camera.H
    window = PygameWindow(W, H, scene_xml.split('\\')[-1].split('/')[-1].removesuffix('.xml'))
    renderer = Renderer(scene, settings, window.update)
    renderer.render()
    window.wait()