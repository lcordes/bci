import pygame
from pygame import Rect
import zmq
from random import randrange

TIMESTEP = 100
WINDOW_SIZE = 600


class Game:
    def __init__(self, timestep, w_size):
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:5555")
            print("Connected to BCI server")

        except:
            print("Couldn't connect to BCI server")

        pygame.init()
        self.timestep = timestep
        self.w_size = w_size
        self.window = pygame.display.set_mode((self.w_size, self.w_size))
        pygame.display.set_caption("Ladders")
        self.rect_size = WINDOW_SIZE // 10
        self.player = Rect(
            ((self.w_size / 2), (self.w_size - self.rect_size)),
            (self.rect_size, self.rect_size),
        )

        # Generate floors
        gaps = [randrange(0, self.w_size, self.rect_size) for _ in range(5)]
        self.floors = []
        for i in range(5):
            rect1 = Rect(0, 2 * i * self.rect_size, gaps[i], self.rect_size)
            rect2 = Rect(
                gaps[i] + self.rect_size,
                2 * i * self.rect_size,
                self.w_size - gaps[i] - self.rect_size,
                self.rect_size,
            )
            self.floors.extend([rect1, rect2])

        self.running = True

    def run(self):
        while self.running:
            pygame.time.delay(self.timestep)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break

            self.socket.send(b"Command")
            command = self.socket.recv().decode("UTF-8")

            if command == "left":
                new_pos = self.player.move(-self.rect_size, 0)

            if command == "right":
                new_pos = self.player.move(self.rect_size, 0)

            if command == "up":
                new_pos = self.player.move(0, -self.rect_size)

            if (
                new_pos.collidelist(self.floors) == -1
                and 0 <= new_pos.left < self.w_size
            ):
                self.player = new_pos

            if self.player.top <= 0:
                print("You win!")
                self.running = False

            # Redraw screen
            self.window.fill((255, 255, 102))
            for floor in self.floors:
                pygame.draw.rect(self.window, (210, 105, 30), floor)
            pygame.draw.rect(
                self.window,
                (255, 51, 51),
                (self.player.left, self.player.top, self.rect_size, self.rect_size),
            )
            pygame.display.update()

        print("Game was ended.")
        pygame.quit()


if __name__ == "__main__":
    sandbox = Game(TIMESTEP, WINDOW_SIZE)
    sandbox.run()
