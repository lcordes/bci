import pygame
from pygame import Rect
from bci_client import BCIClient


WINDOW_SIZE = 600
TIMESTEP = 100


RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
BACKGROUND = (255, 255, 102)


class Bar:
    def __init__(self, timestep, w_size):
        self.client = BCIClient()
        pygame.init()
        self.timestep = timestep
        self.w_size = w_size
        self.window = pygame.display.set_mode((self.w_size, self.w_size))
        pygame.display.set_caption("Concentration")
        self.height = (self.w_size // 3) * 2
        self.width = (self.w_size // 3) * 1
        self.border = Rect(
            (self.w_size // 3),
            (self.w_size // 6),
            self.width,
            self.height,
        )

    def redraw_screen(self):
        self.window.fill(BACKGROUND)
        bar_length = self.concentration * (self.height // 100)
        bar = Rect(
            (self.w_size // 3),
            ((self.height + self.w_size // 6) - bar_length),
            self.width,
            bar_length,
        )
        pygame.draw.rect(self.window, BLUE, self.border, 5)
        pygame.draw.rect(self.window, BLUE, bar)

        pygame.display.update()

    def end_session(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def run(self):
        while True:

            self.client.request_command()
            self.concentration = int(self.client.get_command())
            self.redraw_screen()

            if self.end_session():
                break

            pygame.time.delay(self.timestep)


if __name__ == "__main__":
    bar = Bar(TIMESTEP, WINDOW_SIZE)
    bar.run()
