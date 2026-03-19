#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define MAX_SIZE 6
#define UP 1
#define DOWN 2
#define RIGHT 3
#define LEFT 4

typedef enum {
    MENU,
    PLAYING,
    HELP,
    GAME_OVER,
    GAME_WON
} GameState;

typedef struct {
    int size;
    int table[MAX_SIZE][MAX_SIZE];
    int score;
    GameState state;
    int difficulty;
} Game;

// Function prototypes
void InitGame(Game* game, int difficulty);
int RandomNumber();
void AddRandomTile(Game* game);
void SwapUp(Game* game);
void SwapDown(Game* game);
void SwapRight(Game* game);
void SwapLeft(Game* game);
int TerminatingCondition(Game* game);
int WinningCondition(Game* game);
void DrawGame(Game* game, int screenWidth, int screenHeight);
void DrawMenu(int screenWidth, int screenHeight);
void DrawHelp(int screenWidth, int screenHeight);
void DrawGameOver(Game* game, int screenWidth, int screenHeight);
Color GetTileColor(int value);
Color GetTextColor(int value);

int main(void) {
    const int initialWidth = 800;
    const int initialHeight = 600;
    
    InitWindow(initialWidth, initialHeight, "2048 Game");
    SetWindowState(FLAG_WINDOW_RESIZABLE);
    SetTargetFPS(60);
    
    Game game = {0};
    game.state = MENU;
    
    srand(time(NULL));
    
    while (!WindowShouldClose()) {
        int screenWidth = GetScreenWidth();
        int screenHeight = GetScreenHeight();
        
        // Input handling based on state
        switch (game.state) {
            case MENU:
                if (IsKeyPressed(KEY_ONE)) {
                    InitGame(&game, 1);
                    game.state = PLAYING;
                }
                else if (IsKeyPressed(KEY_TWO)) {
                    InitGame(&game, 2);
                    game.state = PLAYING;
                }
                else if (IsKeyPressed(KEY_THREE)) {
                    InitGame(&game, 3);
                    game.state = PLAYING;
                }
                else if (IsKeyPressed(KEY_H)) {
                    game.state = HELP;
                }
                break;
                
            case PLAYING: {
                int moved = 0;
                int temp[MAX_SIZE][MAX_SIZE];
                
                // Copy current state
                for (int i = 0; i < game.size; i++)
                    for (int j = 0; j < game.size; j++)
                        temp[i][j] = game.table[i][j];
                
                if (IsKeyPressed(KEY_UP) || IsKeyPressed(KEY_W)) {
                    SwapUp(&game);
                    moved = 1;
                }
                else if (IsKeyPressed(KEY_DOWN) || IsKeyPressed(KEY_S)) {
                    SwapDown(&game);
                    moved = 1;
                }
                else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) {
                    SwapRight(&game);
                    moved = 1;
                }
                else if (IsKeyPressed(KEY_LEFT) || IsKeyPressed(KEY_A)) {
                    SwapLeft(&game);
                    moved = 1;
                }
                else if (IsKeyPressed(KEY_H)) {
                    game.state = HELP;
                }
                else if (IsKeyPressed(KEY_ESCAPE)) {
                    game.state = MENU;
                }
                
                // Check if board changed
                if (moved) {
                    int changed = 0;
                    for (int i = 0; i < game.size; i++) {
                        for (int j = 0; j < game.size; j++) {
                            if (temp[i][j] != game.table[i][j]) {
                                changed = 1;
                                break;
                            }
                        }
                        if (changed) break;
                    }
                    
                    if (changed) {
                        AddRandomTile(&game);
                        
                        // Check win condition
                        if (WinningCondition(&game)) {
                            game.state = GAME_WON;
                        }
                        // Check lose condition
                        else if (!TerminatingCondition(&game)) {
                            game.state = GAME_OVER;
                        }
                    }
                }
                break;
            }
            
            case HELP:
                if (IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_ENTER)) {
                    game.state = (game.size > 0) ? PLAYING : MENU;
                }
                break;
                
            case GAME_OVER:
            case GAME_WON:
                if (IsKeyPressed(KEY_ENTER)) {
                    InitGame(&game, game.difficulty);
                    game.state = PLAYING;
                }
                else if (IsKeyPressed(KEY_ESCAPE)) {
                    game.state = MENU;
                }
                break;
        }
        
        // Drawing
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        switch (game.state) {
            case MENU:
                DrawMenu(screenWidth, screenHeight);
                break;
            case PLAYING:
                DrawGame(&game, screenWidth, screenHeight);
                break;
            case HELP:
                DrawHelp(screenWidth, screenHeight);
                break;
            case GAME_OVER:
            case GAME_WON:
                DrawGameOver(&game, screenWidth, screenHeight);
                break;
        }
        
        EndDrawing();
    }
    
    CloseWindow();
    return 0;
}

void InitGame(Game* game, int difficulty) {
    game->difficulty = difficulty;
    game->score = 0;
    
    switch (difficulty) {
        case 1: game->size = 6; break;
        case 2: game->size = 5; break;
        case 3: game->size = 4; break;
        default: game->size = 4; break;
    }
    
    // Clear board
    for (int i = 0; i < MAX_SIZE; i++)
        for (int j = 0; j < MAX_SIZE; j++)
            game->table[i][j] = 0;
    
    // Add two initial tiles
    AddRandomTile(game);
    AddRandomTile(game);
}

int RandomNumber() {
    return (rand() % 10 == 0) ? 4 : 2;
}

void AddRandomTile(Game* game) {
    int empty[MAX_SIZE * MAX_SIZE][2];
    int count = 0;
    
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            if (game->table[i][j] == 0) {
                empty[count][0] = i;
                empty[count][1] = j;
                count++;
            }
        }
    }
    
    if (count > 0) {
        int idx = rand() % count;
        game->table[empty[idx][0]][empty[idx][1]] = RandomNumber();
    }
}

void SwapUp(Game* game) {
    for (int j = 0; j < game->size; j++) {
        int lastMerge = -1;
        for (int i = 1; i < game->size; i++) {
            if (game->table[i][j] == 0) continue;
            
            int k = i;
            while (k > 0 && game->table[k-1][j] == 0) {
                game->table[k-1][j] = game->table[k][j];
                game->table[k][j] = 0;
                k--;
            }
            
            if (k > 0 && game->table[k-1][j] == game->table[k][j] && lastMerge != k-1) {
                game->table[k-1][j] *= 2;
                game->score += game->table[k-1][j];
                game->table[k][j] = 0;
                lastMerge = k-1;
            }
        }
    }
}

void SwapDown(Game* game) {
    for (int j = 0; j < game->size; j++) {
        int lastMerge = game->size;
        for (int i = game->size - 2; i >= 0; i--) {
            if (game->table[i][j] == 0) continue;
            
            int k = i;
            while (k < game->size - 1 && game->table[k+1][j] == 0) {
                game->table[k+1][j] = game->table[k][j];
                game->table[k][j] = 0;
                k++;
            }
            
            if (k < game->size - 1 && game->table[k+1][j] == game->table[k][j] && lastMerge != k+1) {
                game->table[k+1][j] *= 2;
                game->score += game->table[k+1][j];
                game->table[k][j] = 0;
                lastMerge = k+1;
            }
        }
    }
}

void SwapRight(Game* game) {
    for (int i = 0; i < game->size; i++) {
        int lastMerge = game->size;
        for (int j = game->size - 2; j >= 0; j--) {
            if (game->table[i][j] == 0) continue;
            
            int k = j;
            while (k < game->size - 1 && game->table[i][k+1] == 0) {
                game->table[i][k+1] = game->table[i][k];
                game->table[i][k] = 0;
                k++;
            }
            
            if (k < game->size - 1 && game->table[i][k+1] == game->table[i][k] && lastMerge != k+1) {
                game->table[i][k+1] *= 2;
                game->score += game->table[i][k+1];
                game->table[i][k] = 0;
                lastMerge = k+1;
            }
        }
    }
}

void SwapLeft(Game* game) {
    for (int i = 0; i < game->size; i++) {
        int lastMerge = -1;
        for (int j = 1; j < game->size; j++) {
            if (game->table[i][j] == 0) continue;
            
            int k = j;
            while (k > 0 && game->table[i][k-1] == 0) {
                game->table[i][k-1] = game->table[i][k];
                game->table[i][k] = 0;
                k--;
            }
            
            if (k > 0 && game->table[i][k-1] == game->table[i][k] && lastMerge != k-1) {
                game->table[i][k-1] *= 2;
                game->score += game->table[i][k-1];
                game->table[i][k] = 0;
                lastMerge = k-1;
            }
        }
    }
}

int TerminatingCondition(Game* game) {
    // Check for empty cells
    for (int i = 0; i < game->size; i++)
        for (int j = 0; j < game->size; j++)
            if (game->table[i][j] == 0) return 1;
    
    // Check for possible merges
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            if (j < game->size - 1 && game->table[i][j] == game->table[i][j+1])
                return 1;
            if (i < game->size - 1 && game->table[i][j] == game->table[i+1][j])
                return 1;
        }
    }
    return 0;
}

int WinningCondition(Game* game) {
    for (int i = 0; i < game->size; i++)
        for (int j = 0; j < game->size; j++)
            if (game->table[i][j] == 2048)
                return 1;
    return 0;
}

Color GetTileColor(int value) {
    switch (value) {
        case 0: return (Color){205, 193, 180, 255};
        case 2: return (Color){238, 228, 218, 255};
        case 4: return (Color){237, 224, 200, 255};
        case 8: return (Color){242, 177, 121, 255};
        case 16: return (Color){245, 149, 99, 255};
        case 32: return (Color){246, 124, 95, 255};
        case 64: return (Color){246, 94, 59, 255};
        case 128: return (Color){237, 207, 114, 255};
        case 256: return (Color){237, 204, 97, 255};
        case 512: return (Color){237, 200, 80, 255};
        case 1024: return (Color){237, 197, 63, 255};
        case 2048: return (Color){237, 194, 46, 255};
        default: return (Color){60, 58, 50, 255};
    }
}

Color GetTextColor(int value) {
    return (value <= 4) ? (Color){119, 110, 101, 255} : (Color){249, 246, 242, 255};
}

void DrawGame(Game* game, int screenWidth, int screenHeight) {
    int boardSize = (screenWidth < screenHeight ? screenWidth : screenHeight) - 200;
    int cellSize = boardSize / game->size;
    int padding = 10;
    int startX = (screenWidth - boardSize) / 2;
    int startY = (screenHeight - boardSize) / 2 + 40;
    
    // Draw title and score
    DrawText("2048", screenWidth/2 - 60, 20, 60, DARKGRAY);
    char scoreText[50];
    sprintf(scoreText, "Score: %d", game->score);
    DrawText(scoreText, screenWidth/2 - MeasureText(scoreText, 30)/2, 90, 30, DARKGRAY);
    
    // Draw grid
    for (int i = 0; i < game->size; i++) {
        for (int j = 0; j < game->size; j++) {
            int x = startX + j * cellSize + padding;
            int y = startY + i * cellSize + padding;
            int size = cellSize - padding * 2;
            
            DrawRectangleRounded((Rectangle){x, y, size, size}, 0.1f, 8, GetTileColor(game->table[i][j]));
            
            if (game->table[i][j] != 0) {
                char text[10];
                sprintf(text, "%d", game->table[i][j]);
                int fontSize = (game->table[i][j] < 100) ? 40 : (game->table[i][j] < 1000) ? 35 : 30;
                if (cellSize < 80) fontSize = fontSize * cellSize / 100;
                int textWidth = MeasureText(text, fontSize);
                DrawText(text, x + size/2 - textWidth/2, y + size/2 - fontSize/2, 
                        fontSize, GetTextColor(game->table[i][j]));
            }
        }
    }
    
    // Draw controls
    DrawText("Controls: Arrow Keys or WASD | H: Help | ESC: Menu", 
             screenWidth/2 - 250, screenHeight - 40, 20, GRAY);
}

void DrawMenu(int screenWidth, int screenHeight) {
    DrawText("2048 GAME", screenWidth/2 - 150, 100, 60, DARKGRAY);
    
    DrawText("SELECT DIFFICULTY:", screenWidth/2 - 120, 200, 30, DARKGRAY);
    DrawText("[1] 6x6 (Easy)", screenWidth/2 - 100, 260, 25, GRAY);
    DrawText("[2] 5x5 (Medium)", screenWidth/2 - 100, 300, 25, GRAY);
    DrawText("[3] 4x4 (Hard)", screenWidth/2 - 100, 340, 25, GRAY);
    DrawText("[H] Help", screenWidth/2 - 100, 400, 25, GRAY);
    
    DrawText("Press a number to start", screenWidth/2 - 130, screenHeight - 100, 25, LIGHTGRAY);
}

void DrawHelp(int screenWidth, int screenHeight) {
    int y = 50;
    DrawText("HOW TO PLAY 2048", screenWidth/2 - 180, y, 40, DARKGRAY);
    
    y += 80;
    DrawText("OBJECTIVE:", 50, y, 25, DARKGRAY);
    y += 35;
    DrawText("Create a tile with the number 2048 to win", 50, y, 20, GRAY);
    
    y += 50;
    DrawText("RULES:", 50, y, 25, DARKGRAY);
    y += 35;
    DrawText("- Use arrow keys or WASD to move tiles", 50, y, 20, GRAY);
    y += 30;
    DrawText("- When two tiles with same number touch, they merge", 50, y, 20, GRAY);
    y += 30;
    DrawText("- After each move, a new tile (2 or 4) appears", 50, y, 20, GRAY);
    
    y += 50;
    DrawText("CONTROLS:", 50, y, 25, DARKGRAY);
    y += 35;
    DrawText("Arrow Keys or WASD - Move tiles", 50, y, 20, GRAY);
    y += 30;
    DrawText("H - Show this help screen", 50, y, 20, GRAY);
    y += 30;
    DrawText("ESC - Return to menu", 50, y, 20, GRAY);
    
    DrawText("Press ENTER or ESC to continue", screenWidth/2 - 170, screenHeight - 60, 25, LIGHTGRAY);
}

void DrawGameOver(Game* game, int screenWidth, int screenHeight) {
    DrawGame(game, screenWidth, screenHeight);
    
    // Semi-transparent overlay
    DrawRectangle(0, 0, screenWidth, screenHeight, (Color){0, 0, 0, 150});
    
    if (game->state == GAME_WON) {
        DrawText("YOU WIN!", screenWidth/2 - 150, screenHeight/2 - 60, 60, GOLD);
    } else {
        DrawText("GAME OVER", screenWidth/2 - 180, screenHeight/2 - 60, 60, RED);
    }
    
    char scoreText[50];
    sprintf(scoreText, "Final Score: %d", game->score);
    DrawText(scoreText, screenWidth/2 - MeasureText(scoreText, 30)/2, screenHeight/2 + 20, 30, WHITE);
    
    DrawText("Press ENTER to play again", screenWidth/2 - 150, screenHeight/2 + 80, 25, LIGHTGRAY);
    DrawText("Press ESC for menu", screenWidth/2 - 110, screenHeight/2 + 120, 25, LIGHTGRAY);
}