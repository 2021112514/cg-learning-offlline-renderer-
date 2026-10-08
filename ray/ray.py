from device import Array


class Ray:
    def __init__(
        self,
        direction: Array,
        pos: Array,
        min_t: Array,
        max_t: Array
    ):
        self.direction = direction
        self.pos = pos
        self.min_t = min_t
        self.max_t = max_t