from datetime import datetime
import argparse
import pygame
import pandas as pd
from random import choice
from bci_client import BCIClient

TIMESTEP = 50
WINDOW_SIZE = 600
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class Checker:
    def __init__(self, timestep, w_size, keep_log):
        self.client = BCIClient()
        pygame.init()
        self.keep_log = keep_log
        self.timestep = timestep
        self.w_size = w_size
        self.window = pygame.display.set_mode((self.w_size, self.w_size))
        self.state = "start"
        pygame.display.set_caption("Accuracy Checker")
        self.font = pygame.font.Font("freesansbold.ttf", 26)
        self.trial = 1
        self.log = []

        self.window.fill(WHITE)
        pygame.display.update()

    def end_app(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def display_text(self, string, colour):
        text = self.font.render(string, True, colour)
        textRect = text.get_rect()
        textRect.center = (self.w_size // 2, self.w_size // 2)
        self.window.fill(WHITE)
        self.window.blit(text, textRect)
        pygame.display.update()

    def run(self):
        while True:
            # add pause state
            if self.state == "start":
                # Require spacebar to start recording
                self.display_text("Start recording", BLACK)
                self.state = "arrow"
                pygame.time.delay(2000)

            elif self.state == "arrow":
                self.current_arrow = choice(["left", "right", "up"])
                self.display_text(self.current_arrow, BLACK)  # Draw arrow instead
                self.state = "imagine"
                pygame.time.delay(2000)

            elif self.state == "imagine":
                self.display_text("", BLACK)
                self.state = "feedback"
                pygame.time.delay(3000)

            elif self.state == "feedback":
                self.client.request_command()
                command = self.client.get_command()
                if command == self.current_arrow:
                    self.display_text("Correct Classification", GREEN)
                else:
                    self.display_text(f"Incorrect Classification", RED)

                self.state = "arrow"
                data = {
                    "trial": self.trial,
                    "instruction": self.current_arrow,
                    "prediction": command,
                }
                print(data)
                self.log.append(data)
                self.trial += 1
                pygame.time.delay(2000)  # Correct this for processing time

            if self.end_app():
                if self.keep_log:
                    session_time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
                    data = pd.DataFrame.from_dict(self.log)
                    data.to_csv(f"data/recordings/Accuracy_{session_time}.csv")
                break

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        help="write session data to csv",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    checker = Checker(TIMESTEP, WINDOW_SIZE, keep_log=args.log)
    checker.run()
