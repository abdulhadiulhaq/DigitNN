#include "raylib.h"
#include <iostream>
#include "Network.h"
#include "Matrix.h"
// Raylib defines its own Matrix — use fully qualified name if conflict
int main(){
    const int CELL_SIZE   = 16;
    const int GRID_SIZE   = 28;
    const int CANVAS_SIZE = CELL_SIZE * GRID_SIZE;
    const int SIDEBAR_W   = 200;
    const int WINDOW_H = 468;
    const int WINDOW_W = 648;
    const Color BG_COLOR     = {15,  15,  15,  255};
    const Color CANVAS_BG    = {30,  30,  30,  255};
    const Color SIDEBAR_BG   = {20,  20,  20,  255};
    const Color BUTTON_COLOR = {60,  60,  60,  255};
    const Color BUTTON_HOVER = {90,  90,  90,  255};
    const Color TEXT_COLOR   = {220, 220, 220, 255};
    Network net({ 784 , 128 , 64 , 10 });
    net.loadWeights("../data/weights.txt");
    int prediction = -1;

    double canvas[28][28] = {};
    InitWindow( 648 , 468 , "Digit Recognizing Neural Network" );
    SetTargetFPS( 63 );
    while( !WindowShouldClose() && !IsKeyPressed(KEY_ESCAPE) ){

        Vector2 mouse = GetMousePosition();
        int col = ( int )( mouse.x / CELL_SIZE );
        int row = ( int )( mouse.y / CELL_SIZE );

        if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
            if( col >= 0 && col < 28 && row >= 0 && row < 28 ){
                canvas[ row ][ col ] = 1.0;
                if( row-1 >= 0 )canvas[ row - 1 ][ col ] = 0.5;
                if( row+1 < 28 )canvas[ row + 1 ][ col ] = 0.5;
                if( col-1 >= 0 )canvas[ row ][ col - 1 ] = 1.0;
                if( col+1 < 28 )canvas[ row ][ col + 1 ] = 1.0;
            }
        }

        if (IsMouseButtonDown(MOUSE_RIGHT_BUTTON)) {
            for (int r = 0; r < 28; r++)
                for (int c = 0; c < 28; c++)
                    canvas[r][c] = 0.0;
        }
        BeginDrawing();
        ClearBackground( BG_COLOR);
        DrawRectangle( CANVAS_SIZE , 0 , SIDEBAR_W , WINDOW_H , SIDEBAR_BG);
        for( int r = 0 ; r < 28 ; r++ ){
            for ( int c = 0 ; c < 28 ; c++ ){
                unsigned char brightness = ( unsigned char )(canvas[ r ][ c ] * 255 );
                Color color = { brightness , brightness , brightness , 255 };
                DrawRectangle( c * CELL_SIZE , r * CELL_SIZE , CELL_SIZE - 1 , CELL_SIZE - 1 , color );
                DrawRectangle(0, CANVAS_SIZE, WINDOW_W, 20, {10, 10, 10, 255});
            }
        }
        DrawRectangle( CANVAS_SIZE+50 , 412 , 100 , 34 , RED);
        DrawText("Clear", CANVAS_SIZE+75, 420 , 20, WHITE);
        Rectangle btn = { CANVAS_SIZE+50 , 412 , 100 , 34 };
        if( CheckCollisionPointRec( GetMousePosition() , btn) && IsMouseButtonDown( MOUSE_LEFT_BUTTON ) ){
            for( int r = 0 ; r < 28 ; r++ ){
                for ( int c = 0 ; c < 28 ; c++ ){
                    canvas[ r ][ c ] = 0.0;
                }
            }
        }
        if (IsKeyPressed(KEY_SPACE)){
            bool hasDrawing = false;
            for (int r = 0; r < 28; r++)
                for (int c = 0; c < 28; c++)
                    if (canvas[r][c] > 0) { hasDrawing = true; break; }
            
            if (hasDrawing) {
                NN::Matrix input_pixels(784, 1);
                for (int r = 0; r < 28; r++)
                    for (int c = 0; c < 28; c++)
                        input_pixels.set_element(r * 28 + c, 0, canvas[r][c]);
                
                NN::Matrix output = net.forward(input_pixels);
                double maxVal = -1;
                for (int i = 0; i < 10; i++) {
                    double val = output.get_matrix_element(i, 0);
                    if (val > maxVal) { maxVal = val; prediction = i; }
                }
            }
        }

        DrawText("Draw a digit", CANVAS_SIZE + 30, 70, 20, WHITE);
        DrawText("Right click", CANVAS_SIZE + 35, 95, 20, WHITE);
        DrawText("to clear", CANVAS_SIZE + 45, 120, 20, WHITE);
        DrawText("Prediction:", CANVAS_SIZE + 10, 175, 20, WHITE);
        if( prediction >= 0 ){
            DrawText(TextFormat("%d", prediction ), CANVAS_SIZE + 70, 200, 80, YELLOW);
        }
        else{
            DrawText("?", CANVAS_SIZE + 70, 200, 80, YELLOW);
        }
        EndDrawing();
    }
    CloseWindow();
}