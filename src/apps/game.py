import argparse
import pygame
from pygame import Rect
from random import randrange
from bci_client import BCIClient

TIMESTEP = 50
WINDOW_SIZE = 600


class Sandbox:
    def __init__(self, timestep, w_size):
        self.client = BCIClient()
        pygame.init()
        self.timestep = timestep
        self.w_size = w_size
        self.window = pygame.display.set_mode((self.w_size, self.w_size))
        pygame.display.set_caption("Sandbox")
        self.rect_size = WINDOW_SIZE // 10
        self.start_pos = Rect(
            ((self.w_size / 2), (self.w_size - self.rect_size)),
            (self.rect_size, self.rect_size),
        )
        self.player = self.start_pos

    def get_new_pos(self, command):

        if command == "left":
            new_pos = self.player.move(-self.rect_size, 0)

        if command == "right":
            new_pos = self.player.move(self.rect_size, 0)

        if command == "up":
            new_pos = self.player.move(0, -self.rect_size)
        return new_pos

    def check_new_pos(self, new_pos):
        # Returns the start position if new position would be outside of the game window
        if new_pos.left < 0 or new_pos.left >= self.w_size or new_pos.top < 0:
            return self.start_pos
        return new_pos

    def redraw_screen(self):
        self.window.fill((255, 255, 102))
        pygame.draw.rect(
            self.window,
            (255, 51, 51),
            (self.player.left, self.player.top, self.rect_size, self.rect_size),
        )
        pygame.display.update()

    def game_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def run(self):
        while True:
            pygame.time.delay(self.timestep)

            self.client.request_command()
            command = self.client.get_command()
            new_pos = self.get_new_pos(command)
            self.player = self.check_new_pos(new_pos)
            self.redraw_screen()

            if self.game_over():
                break

        pygame.quit()


class Ladders(Sandbox):
    def __init__(self, timestep, w_size):
        super().__init__(timestep, w_size)
        pygame.display.set_caption("Ladders")

        # Randomly choose gaps per floor and then get obstacles left and right of gap
        self.gaps = [randrange(0, self.w_size, self.rect_size) for _ in range(5)]
        self.obstacles = []
        for floor in range(5):
            new_obstacles = self.get_obstacles(floor)
            self.obstacles.extend(new_obstacles)

    def check_new_pos(self, new_pos):
        # Returns the current position if new position would collide with obstacles or exit the game window
        if (
            new_pos.collidelist(self.obstacles) == -1
            and 0 <= new_pos.left < self.w_size
        ):
            return new_pos
        return self.player

    def redraw_screen(self):
        self.window.fill((255, 255, 102))
        for obstacle in self.obstacles:
            pygame.draw.rect(self.window, (210, 105, 30), obstacle)
        pygame.draw.rect(
            self.window,
            (255, 51, 51),
            (self.player.left, self.player.top, self.rect_size, self.rect_size),
        )
        pygame.display.update()

    def get_obstacles(self, floor):
        left_obstacle = Rect(
            0, 2 * floor * self.rect_size, self.gaps[floor], self.rect_size
        )
        right_obstacle = Rect(
            self.gaps[floor] + self.rect_size,
            2 * floor * self.rect_size,
            self.w_size - self.gaps[floor] - self.rect_size,
            self.rect_size,
        )
        return [left_obstacle, right_obstacle]

    def game_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        if self.player.top < 0:
            print("You win!")
            return True
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sandbox",
        help="start game in sandbox mode",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    game = (
        Sandbox(TIMESTEP, WINDOW_SIZE)
        if args.sandbox
        else Ladders(TIMESTEP, WINDOW_SIZE)
    )
    game.run()
