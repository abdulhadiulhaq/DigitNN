#include "raylib.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <string>
#include <vector>
#include <array>

constexpr int MAX_SIZE = 6;

enum Direction { UP = 1, DOWN, RIGHT, LEFT };

enum class GameState {
    MENU,
    PLAYING,
    HELP,
    GAME_OVER,
    GAME_WON
};

struct Game {
    int size = 4;
    std::array<std::array<int, MAX_SIZE>, MAX_SIZE> table{};
    int score = 0;
    GameState state = GameState::MENU;
    int difficulty = 2;

    void init(int diff);
    void addRandomTile();
    void swapUp();
    void swapDown();
    void swapRight();
    void swapLeft();
    bool terminatingCondition() const;
    bool winningCondition() const;
    void draw(int screenWidth, int screenHeight) const;
    void drawGameOver(int screenWidth, int screenHeight) const;
};

// â”€â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

static int randomTileValue() {
    return (rand() % 10 == 0) ? 4 : 2;
}

static Color getTileColor(int value) {
    switch (value) {
        case 0:    return {205, 193, 180, 255};
        case 2:    return {238, 228, 218, 255};
        case 4:    return {237, 224, 200, 255};
        case 8:    return {242, 177, 121, 255};
        case 16:   return {245, 149,  99, 255};
        case 32:   return {246, 124,  95, 255};
        case 64:   return {246,  94,  59, 255};
        case 128:  return {237, 207, 114, 255};
        case 256:  return {237, 204,  97, 255};
        case 512:  return {237, 200,  80, 255};
        case 1024: return {237, 197,  63, 255};
        case 2048: return {237, 194,  46, 255};
        default:   return { 60,  58,  50, 255};
    }
}

static Color getTextColor(int value) {
    return (value <= 4) ? Color{119, 110, 101, 255} : Color{249, 246, 242, 255};
}

// â”€â”€â”€ Game methods â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

void Game::init(int diff) {
    difficulty = diff;
    score = 0;

    switch (diff) {
        case 1: size = 6; break;
        case 2: size = 5; break;
        case 3: size = 4; break;
        default: size = 4; break;
    }

    for (auto& row : table)
        row.fill(0);

    addRandomTile();
    addRandomTile();
}

void Game::addRandomTile() {
    std::vector<std::pair<int,int>> empty;
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (table[i][j] == 0)
                empty.emplace_back(i, j);

    if (!empty.empty()) {
        auto [r, c] = empty[rand() % static_cast<int>(empty.size())];
        table[r][c] = randomTileValue();
    }
}

void Game::swapUp() {
    for (int j = 0; j < size; ++j) {
        int lastMerge = -1;
        for (int i = 1; i < size; ++i) {
            if (table[i][j] == 0) continue;
            int k = i;
            while (k > 0 && table[k-1][j] == 0) {
                table[k-1][j] = table[k][j];
                table[k][j] = 0;
                --k;
            }
            if (k > 0 && table[k-1][j] == table[k][j] && lastMerge != k-1) {
                table[k-1][j] *= 2;
                score += table[k-1][j];
                table[k][j] = 0;
                lastMerge = k-1;
            }
        }
    }
}

void Game::swapDown() {
    for (int j = 0; j < size; ++j) {
        int lastMerge = size;
        for (int i = size - 2; i >= 0; --i) {
            if (table[i][j] == 0) continue;
            int k = i;
            while (k < size - 1 && table[k+1][j] == 0) {
                table[k+1][j] = table[k][j];
                table[k][j] = 0;
                ++k;
            }
            if (k < size - 1 && table[k+1][j] == table[k][j] && lastMerge != k+1) {
                table[k+1][j] *= 2;
                score += table[k+1][j];
                table[k][j] = 0;
                lastMerge = k+1;
            }
        }
    }
}

void Game::swapRight() {
    for (int i = 0; i < size; ++i) {
        int lastMerge = size;
        for (int j = size - 2; j >= 0; --j) {
            if (table[i][j] == 0) continue;
            int k = j;
            while (k < size - 1 && table[i][k+1] == 0) {
                table[i][k+1] = table[i][k];
                table[i][k] = 0;
                ++k;
            }
            if (k < size - 1 && table[i][k+1] == table[i][k] && lastMerge != k+1) {
                table[i][k+1] *= 2;
                score += table[i][k+1];
                table[i][k] = 0;
                lastMerge = k+1;
            }
        }
    }
}

void Game::swapLeft() {
    for (int i = 0; i < size; ++i) {
        int lastMerge = -1;
        for (int j = 1; j < size; ++j) {
            if (table[i][j] == 0) continue;
            int k = j;
            while (k > 0 && table[i][k-1] == 0) {
                table[i][k-1] = table[i][k];
                table[i][k] = 0;
                --k;
            }
            if (k > 0 && table[i][k-1] == table[i][k] && lastMerge != k-1) {
                table[i][k-1] *= 2;
                score += table[i][k-1];
                table[i][k] = 0;
                lastMerge = k-1;
            }
        }
    }
}

bool Game::terminatingCondition() const {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (table[i][j] == 0) return true;

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j) {
            if (j < size - 1 && table[i][j] == table[i][j+1]) return true;
            if (i < size - 1 && table[i][j] == table[i+1][j]) return true;
        }
    return false;
}

bool Game::winningCondition() const {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (table[i][j] == 2048) return true;
    return false;
}

void Game::draw(int screenWidth, int screenHeight) const {
    int boardSize = (screenWidth < screenHeight ? screenWidth : screenHeight) - 200;
    int cellSize  = boardSize / size;
    int padding   = 10;
    int startX    = (screenWidth  - boardSize) / 2;
    int startY    = (screenHeight - boardSize) / 2 + 40;

    DrawText("2048", screenWidth/2 - 60, 20, 60, DARKGRAY);

    char scoreText[64];
    std::snprintf(scoreText, sizeof(scoreText), "Score: %d", score);
    DrawText(scoreText, screenWidth/2 - MeasureText(scoreText, 30)/2, 90, 30, DARKGRAY);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int x    = startX + j * cellSize + padding;
            int y    = startY + i * cellSize + padding;
            int sz   = cellSize - padding * 2;
            int val  = table[i][j];

            DrawRectangleRounded({(float)x, (float)y, (float)sz, (float)sz},
                                 0.1f, 8, getTileColor(val));

            if (val != 0) {
                char text[16];
                std::snprintf(text, sizeof(text), "%d", val);
                int fontSize = (val < 100) ? 40 : (val < 1000) ? 35 : 30;
                if (cellSize < 80) fontSize = fontSize * cellSize / 100;
                int tw = MeasureText(text, fontSize);
                DrawText(text, x + sz/2 - tw/2, y + sz/2 - fontSize/2,
                         fontSize, getTextColor(val));
            }
        }
    }

    DrawText("Controls: Arrow Keys or WASD | H: Help | ESC: Menu",
             screenWidth/2 - 250, screenHeight - 40, 20, GRAY);
}

void Game::drawGameOver(int screenWidth, int screenHeight) const {
    draw(screenWidth, screenHeight);

    DrawRectangle(0, 0, screenWidth, screenHeight, {0, 0, 0, 150});

    if (state == GameState::GAME_WON)
        DrawText("YOU WIN!", screenWidth/2 - 150, screenHeight/2 - 60, 60, GOLD);
    else
        DrawText("GAME OVER", screenWidth/2 - 180, screenHeight/2 - 60, 60, RED);

    char scoreText[64];
    std::snprintf(scoreText, sizeof(scoreText), "Final Score: %d", score);
    DrawText(scoreText, screenWidth/2 - MeasureText(scoreText, 30)/2,
             screenHeight/2 + 20, 30, WHITE);

    DrawText("Press ENTER to play again", screenWidth/2 - 150, screenHeight/2 + 80,  25, LIGHTGRAY);
    DrawText("Press ESC for menu",        screenWidth/2 - 110, screenHeight/2 + 120, 25, LIGHTGRAY);
}

// â”€â”€â”€ Free functions for non-game screens â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

static void drawMenu(int screenWidth, int screenHeight) {
    DrawText("2048 GAME", screenWidth/2 - 150, 100, 60, DARKGRAY);
    DrawText("SELECT DIFFICULTY:", screenWidth/2 - 120, 200, 30, DARKGRAY);
    DrawText("[1] 6x6 (Easy)",   screenWidth/2 - 100, 260, 25, GRAY);
    DrawText("[2] 5x5 (Medium)", screenWidth/2 - 100, 300, 25, GRAY);
    DrawText("[3] 4x4 (Hard)",   screenWidth/2 - 100, 340, 25, GRAY);
    DrawText("[H] Help",         screenWidth/2 - 100, 400, 25, GRAY);
    DrawText("Press a number to start", screenWidth/2 - 130, screenHeight - 100, 25, LIGHTGRAY);
}

static void drawHelp(int screenWidth, int screenHeight) {
    int y = 50;
    DrawText("HOW TO PLAY 2048", screenWidth/2 - 180, y, 40, DARKGRAY);

    y += 80; DrawText("OBJECTIVE:", 50, y, 25, DARKGRAY);
    y += 35; DrawText("Create a tile with the number 2048 to win", 50, y, 20, GRAY);

    y += 50; DrawText("RULES:", 50, y, 25, DARKGRAY);
    y += 35; DrawText("- Use arrow keys or WASD to move tiles", 50, y, 20, GRAY);
    y += 30; DrawText("- When two tiles with same number touch, they merge", 50, y, 20, GRAY);
    y += 30; DrawText("- After each move, a new tile (2 or 4) appears", 50, y, 20, GRAY);

    y += 50; DrawText("CONTROLS:", 50, y, 25, DARKGRAY);
    y += 35; DrawText("Arrow Keys or WASD - Move tiles", 50, y, 20, GRAY);
    y += 30; DrawText("H - Show this help screen", 50, y, 20, GRAY);
    y += 30; DrawText("ESC - Return to menu", 50, y, 20, GRAY);

    DrawText("Press ENTER or ESC to continue",
             screenWidth/2 - 170, screenHeight - 60, 25, LIGHTGRAY);
}

// â”€â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

int main() {
    InitWindow(800, 600, "2048 Game");
    SetWindowState(FLAG_WINDOW_RESIZABLE);
    SetTargetFPS(60);

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    Game game;

    while (!WindowShouldClose()) {
        const int sw = GetScreenWidth();
        const int sh = GetScreenHeight();

        // â”€â”€ Input â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        switch (game.state) {

        case GameState::MENU:
            if      (IsKeyPressed(KEY_ONE))   { game.init(1); game.state = GameState::PLAYING; }
            else if (IsKeyPressed(KEY_TWO))   { game.init(2); game.state = GameState::PLAYING; }
            else if (IsKeyPressed(KEY_THREE)) { game.init(3); game.state = GameState::PLAYING; }
            else if (IsKeyPressed(KEY_H))     { game.state = GameState::HELP; }
            break;

        case GameState::PLAYING: {
            // Snapshot the board to detect whether a move changed anything
            auto snapshot = game.table;

            bool moved = false;
            if      (IsKeyPressed(KEY_UP)    || IsKeyPressed(KEY_W)) { game.swapUp();    moved = true; }
            else if (IsKeyPressed(KEY_DOWN)  || IsKeyPressed(KEY_S)) { game.swapDown();  moved = true; }
            else if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_D)) { game.swapRight(); moved = true; }
            else if (IsKeyPressed(KEY_LEFT)  || IsKeyPressed(KEY_A)) { game.swapLeft();  moved = true; }
            else if (IsKeyPressed(KEY_H))      { game.state = GameState::HELP;   break; }
            else if (IsKeyPressed(KEY_ESCAPE)) { game.state = GameState::MENU;   break; }

            if (moved && game.table != snapshot) {
                game.addRandomTile();
                if      (game.winningCondition())    game.state = GameState::GAME_WON;
                else if (!game.terminatingCondition()) game.state = GameState::GAME_OVER;
            }
            break;
        }

        case GameState::HELP:
            if (IsKeyPressed(KEY_ESCAPE) || IsKeyPressed(KEY_ENTER))
                game.state = (game.size > 0) ? GameState::PLAYING : GameState::MENU;
            break;

        case GameState::GAME_OVER:
        case GameState::GAME_WON:
            if      (IsKeyPressed(KEY_ENTER))  { game.init(game.difficulty); game.state = GameState::PLAYING; }
            else if (IsKeyPressed(KEY_ESCAPE)) { game.state = GameState::MENU; }
            break;
        }
        BeginDrawing();
        ClearBackground(RAYWHITE);

        switch (game.state) {
        case GameState::MENU:     drawMenu(sw, sh);            break;
        case GameState::PLAYING:  game.draw(sw, sh);           break;
        case GameState::HELP:     drawHelp(sw, sh);            break;
        case GameState::GAME_OVER:
        case GameState::GAME_WON: game.drawGameOver(sw, sh);   break;
        }

        EndDrawing();
    }

    CloseWindow();
    return 0;
}