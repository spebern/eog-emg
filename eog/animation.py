import time
import sys
import screeninfo
import pygame
import os
from eog.labels import EyeMovement
from eog.utils import gen_labels


def get_screen_width_and_height():
    return 1920, 1080


pygame.init()

# read in images
screen_width, screen_height = get_screen_width_and_height()
arrow = pygame.image.load(os.path.join("eog/images", "arrow.png"))
arrow_right = pygame.transform.rotate(arrow, 270)
arrow_left = pygame.transform.rotate(arrow, 90)
arrow_up = pygame.transform.rotate(arrow, 0)
arrow_down = pygame.transform.rotate(arrow, 180)

blink = pygame.image.load(os.path.join("eog/images", "blink.png"))
blink_height, blink_width = blink.get_size()

v_arrow_height, v_arrow_width = arrow.get_size()
h_arrow_height, h_arrow_width = arrow_left.get_size()
black = (0, 0, 0)

# calculate positions of cues to display
mid_pos = ((screen_width - blink_width) // 2, (screen_height - blink_height) // 2)
up_pos = (screen_width // 2, 0)
down_pos = (screen_width // 2, screen_height - v_arrow_height)
left_pos = (0, (screen_height - h_arrow_height) // 2)
right_pos = (screen_width - h_arrow_width, (screen_height - h_arrow_height) // 2)


class Animation:
    def __init__(self, recorder, trials=0):
        labels = gen_labels(trials)
        self._recorder = recorder
        self._labels = labels
        screen_width, screen_height = get_screen_width_and_height()
        screen = pygame.display.set_mode(
            (screen_width, screen_height), pygame.FULLSCREEN
        )
        screen.fill(black)
        pygame.display.update()
        self.screen = screen

    def run(self):
        for label in self._labels:
            self.run_trial(label)
        time.sleep(2)
        pygame.quit()
        sys.exit()

    def display_label(self, label):
        self.screen.fill(black)

        # decide what to show based on label
        if label == EyeMovement.RELAX:
            self.screen.fill(black)
        elif label == EyeMovement.UP:
            self.screen.blit(arrow_up, up_pos)
        elif label == EyeMovement.DOWN:
            self.screen.blit(arrow_down, down_pos)
        elif label == EyeMovement.RIGHT:
            self.screen.blit(arrow_right, right_pos)
        elif label == EyeMovement.LEFT:
            self.screen.blit(arrow_left, left_pos)
        else:
            err = "invalid label: {}".format(label)
            raise Exception(err)
        pygame.display.update()

    def run_trial(self, label):
        # start relaxed (black screen)
        self.screen.fill(black)
        pygame.display.update()

        # start recording while relaxed (black screen)
        self._recorder.record_label(label)
        time.sleep(0.5)

        self.display_label(label)

        # show label
        time.sleep(1.0)

        # go back to relaxed (black screen)
        self.screen.fill(black)
        pygame.display.update()
        time.sleep(0.5)

        # in total we seel 1.1 s, so that recording (which takes 1s) has enough time to finish
