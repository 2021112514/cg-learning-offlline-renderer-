from dataclasses import dataclass
import xml.sax
from parser import XMLHandler, tag


@dataclass
@tag()
class Screen:
    w: int = 640
    h: int = 480


@dataclass
@tag()
class Render:
    ssaa_sample_times: int = 0
    monte_carlo_integration_sample_times: int = 8
    min_reflection_time: int = 1
    max_reflection_time: int = 2
    reflection_probably: float = 0.8


@dataclass
@tag()
class Settings:
    screen: Screen
    render: Render


def get_settings(path: str) -> Settings:
    parser = xml.sax.make_parser()
    handler = XMLHandler()
    parser.setContentHandler(handler)
    parser.parse(path)
    return handler.root