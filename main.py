"""A polished Snake game implemented with pygame.

This module provides an executable entry point that launches a colourful
version of the classic Snake game.  It is designed to be self-contained so
that running ``python main.py`` immediately starts the game.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import pygame


# Screen configuration
CELL_SIZE = 24
GRID_WIDTH = 30  # results in 720px wide window
GRID_HEIGHT = 24  # results in 576px tall window
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 12

# Colours
COLOR_BACKGROUND = (15, 18, 30)
COLOR_GRID = (29, 34, 50)
COLOR_FOOD = (255, 107, 129)
COLOR_TEXT = (240, 240, 255)
COLOR_TEXT_SUBTLE = (170, 170, 200)

DIRECTIONS = {
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
}


@dataclass(frozen=True)
class Point:
    x: int
    y: int

    def __add__(self, other: Tuple[int, int]) -> "Point":
        dx, dy = other
        return Point(self.x + dx, self.y + dy)


class Snake:
    def __init__(self) -> None:
        start_x = GRID_WIDTH // 2
        start_y = GRID_HEIGHT // 2
        self._segments: List[Point] = [
            Point(start_x - i, start_y) for i in range(4)
        ]
        self._direction = (1, 0)
        self._pending_direction = self._direction

    @property
    def head(self) -> Point:
        return self._segments[0]

    @property
    def segments(self) -> List[Point]:
        return list(self._segments)

    def next_head(self) -> Point:
        return self.head + self._pending_direction

    def set_direction(self, new_direction: Tuple[int, int]) -> None:
        # Prevent the snake from reversing directly into itself.
        if len(self._segments) > 1:
            opposite = (-self._direction[0], -self._direction[1])
            if new_direction == opposite:
                return
        self._pending_direction = new_direction

    def move(self, grow: bool = False) -> None:
        self._direction = self._pending_direction
        new_head = self.head + self._direction
        self._segments.insert(0, new_head)
        if not grow:
            self._segments.pop()

    def hits_self(self) -> bool:
        return self.head in self._segments[1:]

    def hits_wall(self) -> bool:
        return not (0 <= self.head.x < GRID_WIDTH and 0 <= self.head.y < GRID_HEIGHT)


class Food:
    def __init__(self) -> None:
        self.position = Point(0, 0)

    def relocate(self, occupied: List[Point]) -> None:
        available = {
            (x, y)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
        } - {(p.x, p.y) for p in occupied}
        if not available:
            return
        self.position = Point(*random.choice(tuple(available)))


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Elegant Snake")
        self.surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 24)
        self.reset()

    def reset(self) -> None:
        self.snake = Snake()
        self.food = Food()
        self.food.relocate(self.snake.segments)
        self.score = 0
        self.game_over = False

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            if not self.game_over:
                self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in DIRECTIONS:
                    self.snake.set_direction(DIRECTIONS[event.key])
                elif self.game_over and event.key == pygame.K_r:
                    self.reset()
        return True

    def update(self) -> None:
        will_eat = self.snake.next_head() == self.food.position
        self.snake.move(grow=will_eat)
        if self.snake.hits_wall() or self.snake.hits_self():
            self.game_over = True
            return

        if will_eat:
            self.score += 1
            self.food.relocate(self.snake.segments)

    def draw(self) -> None:
        self.surface.fill(COLOR_BACKGROUND)
        self._draw_grid()
        self._draw_food()
        self._draw_snake()
        self._draw_hud()
        if self.game_over:
            self._draw_game_over()
        pygame.display.flip()

    def _draw_grid(self) -> None:
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.surface, COLOR_GRID, (x, 0), (x, SCREEN_HEIGHT), 1)
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.surface, COLOR_GRID, (0, y), (SCREEN_WIDTH, y), 1)

    def _draw_food(self) -> None:
        rect = pygame.Rect(
            self.food.position.x * CELL_SIZE,
            self.food.position.y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(self.surface, COLOR_FOOD, rect, border_radius=6)

    def _draw_snake(self) -> None:
        segments = self.snake.segments
        total = len(segments)
        for index, segment in enumerate(segments):
            intensity = 200 - int(120 * (index / max(total - 1, 1)))
            color = (intensity, 255 - intensity // 2, 100)
            rect = pygame.Rect(
                segment.x * CELL_SIZE + 2,
                segment.y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4,
            )
            pygame.draw.rect(self.surface, color, rect, border_radius=10)

    def _draw_hud(self) -> None:
        score_text = self.font_large.render(f"Score: {self.score}", True, COLOR_TEXT)
        self.surface.blit(score_text, (20, 10))
        message = (
            "Press R to restart" if self.game_over else "Use arrow keys to glide"
        )
        info_text = self.font_small.render(message, True, COLOR_TEXT_SUBTLE)
        self.surface.blit(info_text, (20, 60))

    def _draw_game_over(self) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 10, 20, 180))
        self.surface.blit(overlay, (0, 0))

        message = self.font_large.render("Game Over", True, COLOR_TEXT)
        score_msg = self.font_small.render(
            f"Final Score: {self.score}", True, COLOR_TEXT_SUBTLE
        )
        prompt_msg = self.font_small.render(
            "Press R to play again", True, COLOR_TEXT_SUBTLE
        )

        rect = message.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
        self.surface.blit(message, rect)
        rect = score_msg.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        self.surface.blit(score_msg, rect)
        rect = prompt_msg.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60)
        )
        self.surface.blit(prompt_msg, rect)


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()
