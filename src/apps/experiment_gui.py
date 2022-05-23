import pygame
from pygame.mixer import music
import os
from dotenv import load_dotenv

load_dotenv()
PRACTICE_TRIALS = int(os.environ["PRACTICE_TRIALS"])
TRIALS_PER_CLASS = int(os.environ["TRIALS_PER_CLASS"])
PRACTICE_END_MARKER = int(os.environ["PRACTICE_END_MARKER"])

DATA_PATH = os.environ["DATA_PATH"]
TRIAL_END_MARKER = int(os.environ["TRIAL_END_MARKER"])
IMAGERY_PERIOD = int(float(os.environ["IMAGERY_PERIOD"]) * 1000)
CLASSES = os.environ["CLASSES"].split(",")

FRONT_COL = (255, 255, 255)  # white
BACK_COL = (0, 0, 0)  # black
GREEN = (0, 255, 0)
RED = (255, 0, 0)
SHAPE_RADIUS = 20
SHAPE_THICKNESS = 3
PYGAME_KEYS = {
    "space": pygame.K_SPACE,
    "esc": pygame.K_ESCAPE,
    "q": pygame.K_q,
}


class ExperimentGUI:
    def __init__(self, keep_log, fullscreen=True):
        pygame.init()
        self.keep_log = keep_log
        if fullscreen:
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.x_size, self.y_size = self.window.get_size()
        else:
            self.x_size = int(os.environ["X_SIZE"])
            self.y_size = int(os.environ["Y_SIZE"])
            self.window = pygame.display.set_mode((self.x_size, self.y_size))
        self.center = (self.x_size // 2, self.y_size // 2)
        self.state = "start"
        self.font = pygame.font.Font("freesansbold.ttf", 26)
        self.trial = 1
        self.log = []
        self.running = True
        self.pause = False

        self.window.fill(BACK_COL)
        pygame.display.update()

    def display_text(self, string, colour=FRONT_COL, loc=None, redraw=True):
        loc = self.center if not loc else loc
        text = self.font.render(string, True, colour)
        textRect = text.get_rect()
        textRect.center = loc
        if redraw:
            self.window.fill(BACK_COL)
        self.window.blit(text, textRect)
        pygame.display.flip()
        # pygame.display.update()

    def display_text_input(self, instructions):
        self.display_text(instructions, loc=(self.center[0], self.center[1] - 30))
        input_string = ""
        finished = False
        while not finished:
            finished, update = self.update_input(input_string)
            if not input_string == update:
                input_string = update
                self.display_text(
                    instructions, loc=(self.center[0], self.center[1] - 30)
                )
                self.display_text(input_string, redraw=False)
            pygame.time.delay(1)
        return input_string

    def display_feedback(self, probs):
        # Determine feedback color per marker
        cols = [FRONT_COL, FRONT_COL, FRONT_COL]
        cols[CLASSES.index(self.current_class)] = GREEN
        if not self.current_class == self.command:
            cols[CLASSES.index(self.command)] = RED

        # Get location per marker
        offset = self.y_size // 10
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
            radius=SHAPE_RADIUS // 3,
        )
        pygame.display.update()

    def play_sound(self, file):
        music.load(f"{DATA_PATH}/assets/sounds/{file}")
        music.set_volume(0.2)
        music.play()
        # while music.get_busy():
        #     pass  # block while sound is playing

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

    def wait_for_space(self, text):
        self.display_text(text)
        while self.check_events() not in ["space", "quit"]:
            pass

    def pause_menu(self):
        self.display_text("Experiment paused. Continue with spacebar and quit with q.")
        while True:
            event = self.check_events()
            if event == "space":
                break
            elif event == "quit":
                self.exit()
                self.running = False
                break

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit()
                self.running = False
                return "quit"
            elif event.type == pygame.KEYDOWN:
                if event.key == PYGAME_KEYS["esc"]:
                    self.state = "pause"
                    return "pause"
                elif event.key == PYGAME_KEYS["space"]:
                    return "space"

                elif event.key == PYGAME_KEYS["q"]:
                    return "quit"
        return None

    def update_input(self, input_string):
        finished = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if not input_string == "":
                        finished = True
                elif event.key == pygame.K_BACKSPACE:
                    input_string = input_string[:-1]
                elif event.key == PYGAME_KEYS["esc"]:
                    finished = True
                    self.running = False
                else:
                    input_string += event.unicode

        return finished, input_string
