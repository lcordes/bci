from datetime import datetime
import argparse
import pygame
import pandas as pd
from random import choice
from bci_client import BCIClient

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parents[1].resolve())
sys.path.append(parent_dir)
from data_api.data_handler import OpenBCIHandler

WINDOW_SIZE = 600
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
SHAPE_RADIUS = 15
PYGAME_KEYS = {"space": pygame.K_SPACE, "esc": pygame.K_ESCAPE}
MARKERS = {"left": 1.0, "right": 2.0, "up": 3.0}


class ExperimentGUI:
    def __init__(self, w_size, keep_log):
        pygame.init()
        self.keep_log = keep_log
        self.w_size = w_size
        self.window = pygame.display.set_mode((self.w_size, self.w_size))
        self.state = "start"
        self.font = pygame.font.Font("freesansbold.ttf", 26)
        self.trial = 1
        self.log = []
        self.running = True
        self.pause = False

        self.window.fill(WHITE)
        pygame.display.update()

    def display_text(self, string, colour):
        text = self.font.render(string, True, colour)
        textRect = text.get_rect()
        textRect.center = (self.w_size // 2, self.w_size // 2)
        self.window.fill(WHITE)
        self.window.blit(text, textRect)
        pygame.display.update()

    def draw_arrow(self):
        m = self.w_size // 2
        l = SHAPE_RADIUS
        s = l * 5 // 6
        if self.current_marker == "up":
            points = [(m, m - l), (m - s, m + l), (m + s, m + l)]
        elif self.current_marker == "left":
            points = [(m - l, m), (m + l, m - s), (m + l, m + s)]
        elif self.current_marker == "right":
            points = [(m + l, m), (m - l, m - s), (m - l, m + s)]

        self.window.fill(WHITE)
        pygame.draw.polygon(self.window, BLACK, points)
        pygame.display.update()

    def draw_circle(self):
        self.window.fill(WHITE)
        pygame.draw.circle(
            self.window,
            BLACK,
            center=(self.w_size // 2, self.w_size // 2),
            radius=SHAPE_RADIUS // 2,
        )
        pygame.display.update()

    def key_pressed(self, key):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == PYGAME_KEYS[key]:
                    return True
        return False

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_and_exit()
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == PYGAME_KEYS["esc"]:
                    self.pause = True


class DataGenerator(ExperimentGUI):
    def __init__(self, w_size, keep_log, sim):
        super().__init__(w_size, keep_log)
        pygame.display.set_caption("Evaluator")
        self.data_handler = OpenBCIHandler(sim=sim)
        if self.data_handler.status == "no_connection":
            print("\n", "No headset connection, use '--train_sim' for simulated data")
            self.running = False

    def save_and_exit(self):
        self.data_handler.save_and_exit("Training")
        if self.keep_log:
            data = pd.DataFrame.from_dict(self.log)
            session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            data.to_csv(f"data/recordings/Training_trials_{session_time}.csv")

    def run(self):
        while self.running:
            if self.pause:
                self.display_text("Press spacebar to resume", BLACK)
                while not self.key_pressed("space"):
                    pass
                self.pause = False
                self.state = "arrow"

            elif self.state == "start":
                self.display_text("Start session with spacebar", BLACK)

                while not self.key_pressed("space"):
                    pass
                self.state = "arrow"

            elif self.state == "fixdot":
                self.draw_circle()
                self.state = "imagine"
                pygame.time.delay(1500)

            elif self.state == "arrow":
                self.current_marker = choice(list(MARKERS.keys()))
                self.draw_arrow()
                self.state = "fixdot"
                pygame.time.delay(2000)

            elif self.state == "imagine":
                self.data_handler.insert_marker(MARKERS[self.current_marker])
                self.display_text("", BLACK)
                self.state = "trial_end"
                pygame.time.delay(3000)

            elif self.state == "trial_end":
                self.draw_circle()
                self.state = "arrow"
                data = {
                    "trial": self.trial,
                    "instruction": self.current_marker,
                }
                print(data)
                self.log.append(data)
                self.trial += 1
                pygame.time.delay(2000)

            self.check_events()

        pygame.quit()


class Evaluator(ExperimentGUI):
    def __init__(self, w_size, keep_log):
        super().__init__(w_size, keep_log)
        pygame.display.set_caption("Evaluator")
        self.client = BCIClient()

    def save_and_exit(self):
        data = pd.DataFrame.from_dict(self.log)
        acc = (data["instruction"] == data["prediction"]).mean().round(2)
        print(f"Session accuracy: {acc}")

        if self.keep_log:
            session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
            data.to_csv(f"data/recordings/Evaluation_trials_{session_time}.csv")

    def run(self):
        while self.running:
            if self.pause:
                self.display_text("Press spacebar to resume", BLACK)
                while not self.key_pressed("space"):
                    pass
                self.pause = False
                self.state = "arrow"

            elif self.state == "start":
                self.display_text("Start session with spacebar", BLACK)

                while not self.key_pressed("space"):
                    pass
                self.state = "arrow"

            elif self.state == "fixdot":
                self.draw_circle()
                self.state = "imagine"
                pygame.time.delay(1500)

            elif self.state == "arrow":
                self.current_marker = choice(["left", "right", "up"])
                self.draw_arrow()
                self.state = "fixdot"
                pygame.time.delay(2000)

            elif self.state == "imagine":
                self.display_text("", BLACK)
                self.state = "feedback"
                pygame.time.delay(3000)

            elif self.state == "feedback":
                self.client.request_command()
                command = self.client.get_command()
                if command == self.current_marker:
                    self.display_text("Correct Classification", GREEN)
                else:
                    self.display_text(f"Incorrect Classification", RED)

                self.state = "arrow"
                data = {
                    "trial": self.trial,
                    "instruction": self.current_marker,
                    "prediction": command,
                }
                print(data)
                self.log.append(data)
                self.trial += 1
                pygame.time.delay(2000)

            self.check_events()

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        help="write session data to csv",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--train",
        help="generate training data for model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--train_sim",
        help="generate simulated training data for model",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    gui = (
        DataGenerator(WINDOW_SIZE, keep_log=args.log, sim=args.train_sim)
        if args.train or args.train_sim
        else Evaluator(WINDOW_SIZE, keep_log=args.log)
    )
    gui.run()
