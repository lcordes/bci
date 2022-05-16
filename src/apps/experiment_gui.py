import pygame
import os
from dotenv import load_dotenv

load_dotenv()
TRIALS_PER_CLASS = int(os.environ["TRIALS_PER_CLASS"])
TRIAL_END_MARKER = int(os.environ["TRIAL_END_MARKER"])
IMAGERY_PERIOD = int(float(os.environ["IMAGERY_PERIOD"]) * 1000)
CLASSES = os.environ["CLASSES"].split(",")
X_SIZE = int(os.environ["X_SIZE"])
Y_SIZE = int(os.environ["Y_SIZE"])

FRONT_COL = (255, 255, 255)  # white
BACK_COL = (0, 0, 0)  # black
GREEN = (0, 255, 0)
RED = (255, 0, 0)
SHAPE_RADIUS = 20
SHAPE_THICKNESS = 3
PYGAME_KEYS = {"space": pygame.K_SPACE, "esc": pygame.K_ESCAPE}


class ExperimentGUI:
    def __init__(self, keep_log):
        pygame.init()
        self.keep_log = keep_log
        self.window = pygame.display.set_mode((X_SIZE, Y_SIZE))
        self.center = (X_SIZE // 2, Y_SIZE // 2)
        self.state = "start"
        self.font = pygame.font.Font("freesansbold.ttf", 26)
        self.trial = 1
        self.log = []
        self.running = True
        self.pause = False

        self.window.fill(BACK_COL)
        pygame.display.update()

    def display_text(self, string, colour, loc=None, redraw=True):
        loc = self.center if not loc else loc
        text = self.font.render(string, True, colour)
        textRect = text.get_rect()
        textRect.center = loc
        if redraw:
            self.window.fill(BACK_COL)
        self.window.blit(text, textRect)
        pygame.display.update()

    def display_feedback(self, probs):
        # Determine feedback color per marker
        cols = [FRONT_COL, FRONT_COL, FRONT_COL]
        cols[CLASSES.index(self.current_class)] = GREEN
        if not self.current_class == self.command:
            cols[CLASSES.index(self.command)] = RED

        # Get location per marker
        offset = Y_SIZE // 10
        locs = [
            (self.center[0] - offset, self.center[1]),
            (self.center[0] + offset, self.center[1]),
            (self.center[0], self.center[1] + offset),
        ]

        for prob, col, loc in zip(probs, cols, locs):
            self.display_text(str(prob), col, loc, redraw=False)

    def draw_arrow(self):
        cx = self.center[0]  # center x-a-xis
        cy = self.center[1]  # center x-a-xis
        l = SHAPE_RADIUS
        s = l * 5 // 6
        if self.current_class == "down":
            points = [(cx, cy + l), (cx - s, cy - l), (cx + s, cy - l)]
        elif self.current_class == "left":
            points = [(cx - l, cy), (cx + l, cy - s), (cx + l, cy + s)]
        elif self.current_class == "right":
            points = [(cx + l, cy), (cx - l, cy - s), (cx - l, cy + s)]

        self.window.fill(BACK_COL)
        pygame.draw.polygon(self.window, FRONT_COL, points)
        pygame.display.update()

    def draw_circle(self):
        self.window.fill(BACK_COL)
        pygame.draw.circle(
            self.window,
            FRONT_COL,
            center=self.center,
            radius=SHAPE_RADIUS // 2,
        )
        pygame.display.update()

    def draw_cross(self):
        rad = SHAPE_RADIUS
        thick = SHAPE_THICKNESS
        self.window.fill(BACK_COL)
        bars = [
            pygame.Rect(
                (self.center[0] - (thick // 2), (self.center[1] - (rad // 2))),
                (thick, rad),
            ),
            pygame.Rect(
                (self.center[0] - (rad // 2), (self.center[1] - (thick // 2))),
                (rad, thick),
            ),
        ]

        for bar in bars:
            pygame.draw.rect(self.window, FRONT_COL, bar)
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
                self.exit()
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == PYGAME_KEYS["esc"]:
                    self.pause = True
