from numpy import ndarray


class Window:
    def update(self, pixels: ndarray):
        raise NotImplementedError()

    def wait(self):
        pass