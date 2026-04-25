"""AI Snake Game Agent using LLM to make strategic decisions.

This module implements a classic Snake game where an AI agent powered by
an LLM (via OpenAI API) decides the snake's next move based on the current
game state.
"""

import os
import random
import json
import pygame
from openai import OpenAI

# --- Game Constants ---
WINDOW_SIZE = 600
GRID_SIZE = 20
CELL_SIZE = WINDOW_SIZE // GRID_SIZE
FPS = 8

COLOR_BG = (15, 15, 15)
COLOR_SNAKE = (50, 205, 50)
COLOR_SNAKE_HEAD = (0, 255, 0)
COLOR_FOOD = (220, 50, 50)
COLOR_TEXT = (200, 200, 200)
COLOR_GRID = (30, 30, 30)

DIRECTIONS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_ai_move(snake: list[tuple], food: tuple, current_direction: str) -> str:
    """Ask the LLM for the next best move given the current game state.

    Args:
        snake: List of (x, y) grid positions, head first.
        food: (x, y) grid position of the food.
        current_direction: Current movement direction of the snake.

    Returns:
        One of 'UP', 'DOWN', 'LEFT', 'RIGHT'.
    """
    head_x, head_y = snake[0]
    food_x, food_y = food

    # Build a simple text representation of nearby cells
    danger = {}
    for d, (dx, dy) in DIRECTIONS.items():
        nx, ny = head_x + dx, head_y + dy
        if (nx, ny) in snake or nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
            danger[d] = "wall/body"
        else:
            danger[d] = "safe"

    prompt = (
        f"You are controlling a snake in a {GRID_SIZE}x{GRID_SIZE} grid game.\n"
        f"Snake head position: ({head_x}, {head_y}). Food position: ({food_x}, {food_y}).\n"
        f"Current direction: {current_direction}. Snake length: {len(snake)}.\n"
        f"Danger map (what is in each adjacent cell): {json.dumps(danger)}.\n"
        f"You CANNOT reverse direction (no {OPPOSITE[current_direction]}).\n"
        f"Choose the single best next move to reach the food while avoiding walls and the snake body.\n"
        f"Respond with ONLY one word: UP, DOWN, LEFT, or RIGHT."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        move = response.choices[0].message.content.strip().upper()
        if move in DIRECTIONS and move != OPPOSITE[current_direction]:
            return move
    except Exception as e:
        print(f"[LLM Error] {e}")

    # Fallback: continue in current direction if safe, else pick a safe turn
    if danger[current_direction] == "safe":
        return current_direction
    for d in DIRECTIONS:
        if d != OPPOSITE[current_direction] and danger[d] == "safe":
            return d
    return current_direction


def spawn_food(snake: list[tuple]) -> tuple:
    """Spawn food at a random empty cell."""
    while True:
        pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if pos not in snake:
            return pos


def draw_game(screen: pygame.Surface, snake: list[tuple], food: tuple, score: int, font: pygame.font.Font):
    """Render the current game state to the screen."""
    screen.fill(COLOR_BG)

    # Draw grid lines
    for x in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, WINDOW_SIZE))
    for y in range(0, WINDOW_SIZE, CELL_SIZE):
        pygame.draw.line(screen, COLOR_GRID, (0, y), (WINDOW_SIZE, y))

    # Draw food
    fx, fy = food
    pygame.draw.rect(screen, COLOR_FOOD, (fx * CELL_SIZE + 2, fy * CELL_SIZE + 2, CELL_SIZE - 4, CELL_SIZE - 4))

    # Draw snake
    for i, (sx, sy) in enumerate(snake):
        color = COLOR_SNAKE_HEAD if i == 0 else COLOR_SNAKE
        pygame.draw.rect(screen, color, (sx * CELL_SIZE + 1, sy * CELL_SIZE + 1, CELL_SIZE - 2, CELL_SIZE - 2))

    # Draw score
    score_surf = font.render(f"Score: {score}  |  Length: {len(snake)}", True, COLOR_TEXT)
    screen.blit(score_surf, (8, 8))

    pygame.display.flip()


def main():
    """Main game loop for the AI Snake agent."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("AI Snake Agent")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 18, bold=True)

    # Initial state
    snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
    direction = "RIGHT"
    food = spawn_food(snake)
    score = 0
    running = True

    print("AI Snake Agent started. Press Q or close window to quit.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # Get AI decision
        direction = get_ai_move(snake, food, direction)
        dx, dy = DIRECTIONS[direction]
        new_head = (snake[0][0] + dx, snake[0][1] + dy)

        # Collision detection
        if (
            new_head in snake
            or new_head[0] < 0 or new_head[0] >= GRID_SIZE
            or new_head[1] < 0 or new_head[1] >= GRID_SIZE
        ):
            print(f"Game Over! Final score: {score}, Length: {len(snake)}")
            running = False
            continue

        snake.insert(0, new_head)

        if new_head == food:
            score += 10
            food = spawn_food(snake)
            print(f"Food eaten! Score: {score}")
        else:
            snake.pop()

        draw_game(screen, snake, food, score, font)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
