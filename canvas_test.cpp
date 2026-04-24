#include "raylib.h"
#undef Matrix  // safety: undefine in case of macro conflict

#include <iostream>
#include "Network.h"  // our Network uses NN::Matrix internally
#include "Matrix.h"   // explicit include for NN::Matrix usage in this file

int main() {
    // LAYOUT CONSTANTS
    // CELL_SIZE:   each pixel cell on screen is 16×16 actual pixels
    // GRID_SIZE:   28×28 cells to match MNIST image dimensions exactly
    // CANVAS_SIZE: 28 * 16 = 448px — the drawable area width and height
    // SIDEBAR_W:   200px panel on the right for prediction display
    // WINDOW dimensions: 648×468 — fits comfortably on 1366×768 screens
    const int CELL_SIZE   = 16;
    const int GRID_SIZE   = 28;
    const int CANVAS_SIZE = CELL_SIZE * GRID_SIZE; // 448
    const int SIDEBAR_W   = 200;
    const int WINDOW_H    = 468;
    const int WINDOW_W    = 648;

    // COLOR PALETTE — dark theme matching MNIST's white-on-black style
    // Using near-black instead of pure black (15,15,15 vs 0,0,0) is easier
    // on the eyes for extended use. Sidebar is slightly darker than canvas
    // to create visual separation without a border.
    const Color BG_COLOR     = {15,  15,  15,  255}; // window background
    const Color CANVAS_BG    = {30,  30,  30,  255}; // canvas background
    const Color SIDEBAR_BG   = {20,  20,  20,  255}; // sidebar panel
    const Color BUTTON_COLOR = {60,  60,  60,  255}; // button normal state
    const Color BUTTON_HOVER = {90,  90,  90,  255}; // button hover state
    const Color TEXT_COLOR   = {220, 220, 220, 255}; // soft white text

    // NETWORK SETUP
    // Architecture must EXACTLY match what was used during training in main.cpp.
    // If sizes differ, loadWeights() will read values into wrong-sized matrices
    // causing bad_array_new_length or assertion failures.
    Network net({784, 128, 64, 10});
    net.loadWeights("../data/weights.txt"); // restore trained weights from disk

    int prediction = -1; // -1 means "no prediction yet" (canvas is empty)

    // CANVAS DATA
    // 28×28 grid of doubles, each in range [0.0, 1.0]
    // 0.0 = black (empty pixel), 1.0 = white (drawn pixel)
    // Initialized to all zeros (empty canvas) with = {}
    // This exact structure matches MNIST: each cell becomes one input neuron.
    double canvas[28][28] = {};

    // WINDOW INITIALIZATION
    // InitWindow must be called before any other Raylib functions.
    // SetTargetFPS(63) caps the frame rate — 60fps is standard for UI,
    // 63 adds a tiny buffer above 60 to avoid frame drops on slightly
    // slower frames without going to the overhead of 120fps.
    InitWindow(648, 468, "Digit Recognizing Neural Network");
    SetTargetFPS(63);

    // MAIN LOOP
    // Runs every frame (~63 times per second) until window is closed or
    // ESC is pressed. Each frame:
    //   1. Handle input (drawing, clearing, predicting)
    //   2. Render everything (canvas, sidebar, buttons, prediction)
    while (!WindowShouldClose() && !IsKeyPressed(KEY_ESCAPE)) {

        // INPUT — DRAWING
        // Get mouse position and convert from screen pixels to canvas grid coords.
        //   col = mouse.x / CELL_SIZE  (horizontal position in grid)
        //   row = mouse.y / CELL_SIZE  (vertical position in grid)
        //
        // When left button is held down and cursor is over the canvas:
        //   - Set the cell under cursor to 1.0 (full white)
        //   - Set 4 neighboring cells to 0.5 (grey) for a softer brush feel
        //     This simulates a slightly larger brush so thin strokes are easier
        //     to draw and more closely resemble MNIST training data style.
        //
        // Note: neighbors use 0.5 not 1.0 to create a natural fade at stroke edges.
        // Left/right neighbors use 1.0 (same as center) for horizontal strokes
        // because horizontal mouse movement is more common — debatable choice.
        Vector2 mouse = GetMousePosition();
        int col = (int)(mouse.x / CELL_SIZE);
        int row = (int)(mouse.y / CELL_SIZE);

        if (IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (col >= 0 && col < 28 && row >= 0 && row < 28) {
                canvas[row][col] = 1.0;                              // center cell: full white
                if (row - 1 >= 0) canvas[row-1][col]   = 0.5;       // above: half
                if (row + 1 < 28) canvas[row+1][col]   = 0.5;       // below: half
                if (col - 1 >= 0) canvas[row][col-1]   = 1.0;       // left: full
                if (col + 1 < 28) canvas[row][col+1]   = 1.0;       // right: full
            }
        }

        // INPUT — RIGHT CLICK TO CLEAR
        // Iterates all 784 cells and resets to 0.0.
        // Also resets prediction to -1 so the sidebar shows "?" again.
        if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            for (int r = 0; r < 28; r++)
                for (int c = 0; c < 28; c++)
                    canvas[r][c] = 0.0;
            prediction = -1;
        }

        // INPUT — CLEAR BUTTON (on-screen button in sidebar)
        // Rectangle btn defines the clickable area.
        // CheckCollisionPointRec tests if the mouse is inside the rectangle.
        // Same clear logic as right-click.
        Rectangle btn = {(float)(CANVAS_SIZE + 50), 412, 100, 34};
        if (CheckCollisionPointRec(GetMousePosition(), btn)
            && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            for (int r = 0; r < 28; r++)
                for (int c = 0; c < 28; c++)
                    canvas[r][c] = 0.0;
            prediction = -1;
        }

        // INPUT — SPACE TO PREDICT
        // When SPACE is pressed, run the neural network on the current canvas.
        //
        // Step 1: Check if canvas has any drawing at all.
        //         If empty, skip prediction (network would guess randomly).
        //
        // Step 2: Flatten 2D canvas into (784×1) Matrix column vector.
        //         canvas[r][c] → input_pixels[r*28 + c]
        //         This is the same flattening used in loadImages() so the
        //         network sees the input in the same format it was trained on.
        //
        // Step 3: Run forward pass → get (10×1) output vector.
        //
        // Step 4: Find argmax — the index with the highest output value.
        //         That index is the network's predicted digit.
        if (IsKeyPressed(KEY_SPACE)) {
            // Check if anything is drawn
            bool hasDrawing = false;
            for (int r = 0; r < 28 && !hasDrawing; r++)
                for (int c = 0; c < 28 && !hasDrawing; c++)
                    if (canvas[r][c] > 0) hasDrawing = true;

            if (hasDrawing) {
                // Flatten canvas to 784×1 input vector
                NN::Matrix input_pixels(784, 1);
                for (int r = 0; r < 28; r++)
                    for (int c = 0; c < 28; c++)
                        input_pixels.set_element(r * 28 + c, 0, canvas[r][c]);

                // Forward pass — get confidence scores for digits 0-9
                NN::Matrix output = net.forward(input_pixels);

                // Argmax — find digit with highest confidence
                double maxVal = -1;
                for (int i = 0; i < 10; i++) {
                    double val = output.get_matrix_element(i, 0);
                    if (val > maxVal) {
                        maxVal = val;
                        prediction = i;
                    }
                }
            }
        }

        // RENDERING — everything between BeginDrawing() and EndDrawing()
        // ClearBackground fills the entire window with the background color,
        // erasing whatever was drawn in the previous frame

        BeginDrawing();
        ClearBackground(BG_COLOR);

        // RENDER — SIDEBAR PANEL
        // Drawn first so canvas cells drawn on top of it appear correctly.
        // Rectangle from x=CANVAS_SIZE to end of window, full height.
        DrawRectangle(CANVAS_SIZE, 0, SIDEBAR_W, WINDOW_H, SIDEBAR_BG);

        // RENDER — CANVAS GRID
        // For each of the 784 cells:
        //   brightness = canvas value * 255 (convert 0.0-1.0 to 0-255)
        //   Color is greyscale: R=G=B=brightness
        //   CELL_SIZE-1 width/height leaves a 1px gap = visible grid lines
        //   Position: c*CELL_SIZE (x), r*CELL_SIZE (y)
        //
        // The bottom bar DrawRectangle is placed OUTSIDE this loop to avoid
        // drawing it 784 times (once per cell). It only needs to draw once.
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                unsigned char brightness = (unsigned char)(canvas[r][c] * 255);
                Color color = {brightness, brightness, brightness, 255};
                DrawRectangle(
                    c * CELL_SIZE, r * CELL_SIZE,
                    CELL_SIZE - 1, CELL_SIZE - 1,
                    color
                );
            }
        }

        // RENDER — BOTTOM STATUS BAR
        // Thin dark bar at the bottom for FPS or future status info.
        // Drawn after the canvas loop — once, not 784 times.
        DrawRectangle(0, CANVAS_SIZE, WINDOW_W, 20, {10, 10, 10, 255});

        
        // RENDER — CLEAR BUTTON in sidebar
        // RED background to make it obvious.
        // Button rectangle matches the collision rectangle used in input handling.
        
        DrawRectangle(CANVAS_SIZE + 50, 412, 100, 34, RED);
        DrawText("Clear", CANVAS_SIZE + 75, 420, 20, WHITE);

        
        // RENDER — INSTRUCTIONS in sidebar
        // Simple text telling the user how to interact.
        
        DrawText("Draw a digit", CANVAS_SIZE + 30,  70, 20, WHITE);
        DrawText("Right click",  CANVAS_SIZE + 35,  95, 20, WHITE);
        DrawText("to clear",     CANVAS_SIZE + 45, 120, 20, WHITE);

        
        // RENDER — PREDICTION DISPLAY
        // If prediction == -1, show "?" (nothing drawn yet or just cleared).
        // Otherwise, show the predicted digit as a large yellow number (font 80).
        // Large font makes the prediction immediately visible at a glance.
        // YELLOW stands out strongly against the dark sidebar background.
        
        DrawText("Prediction:", CANVAS_SIZE + 10, 175, 20, WHITE);
        if (prediction >= 0)
            DrawText(TextFormat("%d", prediction), CANVAS_SIZE + 70, 200, 80, YELLOW);
        else
            DrawText("?", CANVAS_SIZE + 70, 200, 80, YELLOW);

        EndDrawing();
    }
    // CLEANUP
    // CloseWindow frees all Raylib resources and closes the OS window.
    // Always call this before exiting to avoid resource leaks.
    CloseWindow();
    return 0;
}