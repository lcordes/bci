import argparse
import pygame
from pygame import Rect
from random import randrange, choice
import numpy as np
from pygame.mixer import music


import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from communication.client_server import Client

import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]
CLASSES = os.environ["CLASSES"].split(",")


IMAGERY_WINDOW = 1000
X_SIZE = 600
RED = (255, 0, 0)
GREEN = (34, 139, 34)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Sandbox:
    def __init__(self, keyboard):
        self.x_size = X_SIZE
        self.y_size = X_SIZE
        self.keyboard = keyboard
        if not keyboard:
            self.client = Client()
            if self.client.connected():
                print("Connected to BCI server")
            else:
                print("Couldn't connect to BCI server")

        pygame.init()
        self.window = pygame.display.set_mode((self.x_size, self.y_size))
        pygame.display.set_caption("Sandbox")
        self.rect_size = self.x_size // 10
        self.player = Rect(
            ((self.x_size // 2 - (self.rect_size // 2)), 0),
            (self.rect_size, self.rect_size),
        )
        self.start_pos = self.player

    def get_new_pos(self, event):

        if event == "left":
            new_pos = self.player.move(-self.rect_size, 0)

        if event == "right":
            new_pos = self.player.move(self.rect_size, 0)

        if event == "down":
            new_pos = self.player.move(0, self.rect_size)
        return new_pos

    def check_new_pos(self, new_pos):
        # Returns the start position if new position would be outside of the game window
        if (
            new_pos.left < 0
            or new_pos.left >= self.x_size
            or new_pos.top >= self.y_size
        ):
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

    def play_sound(self, file):
        music.load(f"{DATA_PATH}/assets/sounds/{file}")
        music.set_volume(0.5)
        music.play()
        pygame.time.delay(500)

    def game_reaction(self, event):
        if event in ["left", "right", "down"]:
            new_pos = self.get_new_pos(event)
            self.player = self.check_new_pos(new_pos)
        self.redraw_screen()

    def get_event(self, prob_string):
        probs = prob_string.split(";")
        probs = [float(prob) for prob in probs]
        return CLASSES[np.argmax(probs)]

    def start_round(self):
        # play tone
        pass

    def get_key(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_LEFT, pygame.K_a]:
                    return "left"
                elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                    return "right"

                elif event.key in [pygame.K_DOWN, pygame.K_s]:
                    return "down"
        return "0"

    def game_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def run(self):
        self.redraw_screen()
        while True:
            # self.play_sound("on_beep.wav")
            self.start_round()
            pygame.time.delay(IMAGERY_WINDOW)
            # self.play_sound("off_beep.wav")

            if self.keyboard:
                event = self.get_key()
            else:
                command = getattr(self, "target", "")
                self.client.request_command(command)
                event_probs = self.client.get_command()
                event = self.get_event(event_probs)

            pygame.time.delay(1000)
            if event in ["left", "right", "down"]:
                self.game_reaction(event)

            self.redraw_screen()
            pygame.time.delay(2000)

            if self.game_over():
                break

        pygame.quit()


class Ladders(Sandbox):
    def __init__(self, keyboard):
        super().__init__(keyboard)
        pygame.display.set_caption("Ladders")
        self.player = Rect(
            (0, 0),
            (self.rect_size, self.rect_size),
        )

        # Randomly choose gaps per floor and then get obstacles left and right of gap
        self.gaps = self.get_gaps()
        self.obstacles = []
        for floor in range(5):
            new_obstacles = self.get_obstacles(floor)
            self.obstacles.extend(new_obstacles)

    def get_gaps(self):
        gaps = []
        gap = 3 * self.rect_size
        prev_gap = None

        for _ in range(5):
            while gap == prev_gap:
                gap = randrange(0, self.x_size, self.rect_size)
            gaps.append(gap)
            prev_gap = gap
        return gaps

    def check_new_pos(self, new_pos):
        # Returns the current position if new position would collide with obstacles or exit the game window
        if (
            new_pos.collidelist(self.obstacles) == -1
            and 0 <= new_pos.left < self.x_size
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
            0, (2 * floor + 1) * self.rect_size, self.gaps[floor], self.rect_size
        )
        right_obstacle = Rect(
            self.gaps[floor] + self.rect_size,
            (2 * floor + 1) * self.rect_size,
            self.x_size - self.gaps[floor] - self.rect_size,
            self.rect_size,
        )
        return [left_obstacle, right_obstacle]

    def game_over(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        if self.player.top >= self.y_size:
            print("You win!")
            return True
        return False


class Shooter(Sandbox):
    def __init__(self, keyboard):
        super().__init__(keyboard)
        pygame.display.set_caption("Shooter")
        self.font = pygame.font.Font("freesansbold.ttf", 26)
        self.y_size = (self.x_size // 3) * 2
        self.window = pygame.display.set_mode((self.x_size, self.y_size))

        self.shape_size = self.x_size // 10
        self.player_center = (self.x_size // 2, self.y_size // 4)
        self.hits = 0
        self.total = 0

        self.target_pos = {
            "left": (self.player_center[0] - self.x_size // 3, self.player_center[1]),
            "right": (self.player_center[0] + self.x_size // 3, self.player_center[1]),
            "down": (self.player_center[0], self.player_center[1] + self.x_size // 3),
        }

    def redraw_screen(self, target=None, colour=WHITE, direction="0"):
        self.window.fill((255, 255, 102))
        self.draw_player(direction)

        if target:
            self.draw_circle(pos=self.target_pos[self.target], colour=colour)

        if self.total > 0:
            self.display_text(
                f"Hits: {self.hits}/{self.total}",
                loc=(int(self.x_size * 0.8), int(self.y_size * 0.9)),
            )

    def display_text(self, string, loc, colour=BLACK):
        loc = self.center if not loc else loc
        text = self.font.render(string, True, colour)
        textRect = text.get_rect()
        textRect.center = loc
        self.window.blit(text, textRect)
        pygame.display.flip()

    def draw_circle(self, pos, colour=WHITE, radius=None, width=0):
        radius = radius if radius else self.shape_size // 2
        pygame.draw.circle(self.window, colour, center=pos, radius=radius, width=width)
        pygame.display.update()

    def draw_player(self, direction):
        self.draw_circle(pos=self.player_center, colour=BLACK)
        length = (self.shape_size // 5) * 4
        if direction == "left":
            end = (self.player_center[0] - length, self.player_center[1])
        elif direction == "right":
            end = (self.player_center[0] + length, self.player_center[1])
        elif direction == "down":
            end = (self.player_center[0], self.player_center[1] + length)
        else:
            end = (self.player_center[0], self.player_center[1] - length)
        pygame.draw.line(
            self.window, BLACK, start_pos=self.player_center, end_pos=end, width=30
        )
        self.draw_cross()
        pygame.display.update()

    def draw_target(self, colour=BLUE):
        self.draw_circle(pos=self.target_pos[self.target])
        self.draw_circle(
            pos=self.target_pos[self.target], colour=colour, radius=self.shape_size // 6
        )
        self.draw_circle(
            pos=self.target_pos[self.target],
            colour=colour,
            radius=self.shape_size // 2,
            width=self.shape_size // 6,
        )
        pygame.display.update()

    def draw_cross(self):
        rad = int(self.shape_size * 0.6)
        thick = self.shape_size // 10
        bars = [
            pygame.Rect(
                (
                    self.player_center[0] - (thick // 2),
                    (self.player_center[1] - (rad // 2)),
                ),
                (thick, rad),
            ),
            pygame.Rect(
                (
                    self.player_center[0] - (rad // 2),
                    (self.player_center[1] - (thick // 2)),
                ),
                (rad, thick),
            ),
        ]

        for bar in bars:
            pygame.draw.rect(self.window, WHITE, bar)

    def start_round(self):
        self.redraw_screen()
        self.target = choice(["left", "right", "down"])
        self.draw_target()

    def game_reaction(self, event):
        self.redraw_screen(direction=event)
        self.draw_target()
        pygame.time.delay(1000)

        if event == self.target:
            colour = GREEN
            self.hits += 1
        else:
            colour = RED
        self.total += 1
        self.redraw_screen(direction=event)
        self.draw_target(colour=colour)
        pygame.time.delay(2000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["sandbox", "ladders", "shooter"],
        help="Start the game in on of the following modes: [sandbox, ladders, shooter].",
    )
    parser.add_argument(
        "--key",
        help="Use keyboard input instead of BCI.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if args.mode == "sandbox":
        game = Sandbox(args.key)
    elif args.mode == "ladders":
        game = Ladders(args.key)
    else:
        game = Shooter(args.key)

    game.run()
