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
GREEN = (0, 150, 0)
RED = (150, 0, 0)
SHAPE_RADIUS = 40
LINE_SPACE = 45
SHAPE_THICKNESS = 6
PYGAME_KEYS = {
    "space": pygame.K_SPACE,
    "esc": pygame.K_ESCAPE,
    "q": pygame.K_q,
}


class ExperimentGUI:
    def __init__(self, fullscreen=True):
        pygame.init()
        if fullscreen:
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.x_size, self.y_size = self.window.get_size()
        else:
            self.x_size = int(os.environ["X_SIZE"])
            self.y_size = int(os.environ["Y_SIZE"])
            self.window = pygame.display.set_mode((self.x_size, self.y_size))
        self.move = self.y_size // 4
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

    def display_multiline(self, lines, colour=FRONT_COL):

        first_line_y = self.center[1] - (len(lines) // 2) * LINE_SPACE

        for i, line in enumerate(lines):
            loc = (self.center[0], first_line_y + i * LINE_SPACE)
            redraw = True if i == 0 else False
            self.display_text(line, loc=loc, redraw=redraw)

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

    def draw_arrow(self, label, col=FRONT_COL, move=False):
        cx = self.center[0]  # center x-a-xis
        cy = self.center[1]  # center y-a-xis
        l = SHAPE_RADIUS * 2 // 3
        s = l * 5 // 6
        if label == "down":
            if move:
                cy += self.move
            points = [(cx, cy + l), (cx - s, cy - l), (cx + s, cy - l)]
        elif label == "left":
            if move:
                cx -= self.move
            points = [(cx - l, cy), (cx + l, cy - s), (cx + l, cy + s)]
        elif label == "right":
            if move:
                cx += self.move
            points = [(cx + l, cy), (cx - l, cy - s), (cx - l, cy + s)]

        self.window.fill(BACK_COL)
        pygame.draw.polygon(self.window, col, points)
        pygame.display.update()

    def draw_circle(self):
        self.window.fill(BACK_COL)
        pygame.draw.circle(
            self.window,
            FRONT_COL,
            center=self.center,
            radius=SHAPE_RADIUS // 4,
        )
        pygame.display.update()

    def play_sound(self, file):
        music.load(f"{DATA_PATH}/assets/sounds/{file}")
        music.set_volume(0.2)
        music.play()

    def draw_cross(self, col=FRONT_COL, move=False):
        rad = SHAPE_RADIUS
        thick = SHAPE_THICKNESS
        self.window.fill(BACK_COL)
        x = self.center[0]
        y = self.center[1]

        if move:
            step = self.y_size // 4
            if move == "left":
                x -= step
            elif move == "right":
                x += step
            else:
                y += step

        bars = [
            pygame.Rect(
                (x - (thick // 2), (y - (rad // 2))),
                (thick, rad),
            ),
            pygame.Rect(
                (x - (rad // 2), (y - (thick // 2))),
                (rad, thick),
            ),
        ]

        for bar in bars:
            pygame.draw.rect(self.window, col, bar)
        pygame.display.update()

    def wait_for_space(self, text):
        if isinstance(text, list):
            self.display_multiline(text)
        else:
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
                    self.pause = True
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
