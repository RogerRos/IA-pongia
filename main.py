import pygame
import sys
import random
import numpy as np
import os

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Ping Pong")

ball_radius = 15
ball_speed_x = 5
ball_speed_y = 5
ball_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]

paddle_width = 10
paddle_height = 100
paddle_speed = 10

player1_pos = [10, (SCREEN_HEIGHT - paddle_height) // 2]
wall_pos = SCREEN_WIDTH - 20

player1_score = 0
player2_score = 0

font = pygame.font.Font(None, 74)

clock = pygame.time.Clock()

# Paràmetres per a Q-learning
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Inicialitza la Q-taula
q_table_file = 'q_table.npy'
if os.path.exists(q_table_file):
    q_table = np.load(q_table_file)
else:
    q_table = np.zeros((SCREEN_HEIGHT // paddle_speed, SCREEN_HEIGHT // paddle_speed, 2))

save_interval = 10000
iteration = 0

def get_state():
    ball_y = ball_pos[1] // paddle_speed
    paddle_y = player1_pos[1] // paddle_speed
    return int(ball_y), int(paddle_y)

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 1)
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = get_state()
    action = choose_action(state)

    if action == 0 and player1_pos[1] > 0:
        player1_pos[1] -= paddle_speed
    if action == 1 and player1_pos[1] < SCREEN_HEIGHT - paddle_height:
        player1_pos[1] += paddle_speed

    ball_pos[0] += ball_speed_x
    ball_pos[1] += ball_speed_y

    reward = 0

    if ball_pos[1] - ball_radius <= 0 or ball_pos[1] + ball_radius >= SCREEN_HEIGHT:
        ball_speed_y = -ball_speed_y

    if ball_pos[0] - ball_radius <= player1_pos[0] + paddle_width and player1_pos[1] <= ball_pos[1] <= player1_pos[1] + paddle_height:
        ball_speed_x = -ball_speed_x
        reward = 1
    if ball_pos[0] + ball_radius >= wall_pos:
        ball_speed_x = -ball_speed_x

    if ball_pos[0] - ball_radius <= 0:
        player2_score += 1
        ball_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
        ball_speed_x = -ball_speed_x
        reward = -1

    next_state = get_state()
    update_q_table(state, action, reward, next_state)

    iteration += 1
    if iteration % save_interval == 0:
        np.save(q_table_file, q_table)
        print(f"Q-table saved at iteration {iteration}")

    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (player1_pos[0], player1_pos[1], paddle_width, paddle_height))
    pygame.draw.rect(screen, WHITE, (wall_pos, 0, paddle_width, SCREEN_HEIGHT))
    pygame.draw.circle(screen, WHITE, ball_pos, ball_radius)

    score_text = font.render(str(player1_score), True, WHITE)
    screen.blit(score_text, (SCREEN_WIDTH // 4, 10))
    score_text = font.render(str(player2_score), True, WHITE)
    screen.blit(score_text, (3 * SCREEN_WIDTH // 4, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
np.save(q_table_file, q_table)
print("Q-table saved on exit")
sys.exit()
