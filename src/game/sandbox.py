import pygame
import zmq


class Game:
    def __init__(self):
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:5555")
            print("Connected to BCI server")

        except:
            print("Couldn't connect to BCI server")

        pygame.init()
        self.timestep = 1000
        self.window = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Sandbox")

        self.x = 400
        self.y = 400
        self.width = 40
        self.height = 40
        self.vel = 20

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
                self.x -= self.vel

            if command == "right":
                self.x += self.vel

            if command == "up":
                self.y -= self.vel

            if command == "down":
                self.y += self.vel

            self.window.fill((255, 153, 0))  # Fills the screen with colour
            pygame.draw.rect(
                self.window, (0, 0, 153), (self.x, self.y, self.width, self.height)
            )
            pygame.display.update()

        print("Game was ended.")
        pygame.quit()


if __name__ == "__main__":
    sandbox = Game()
    sandbox.run()
