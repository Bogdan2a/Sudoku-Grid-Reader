#include <stdio.h>

#define ROWS 9
#define COLS 9

int main() {
    int sudoku[ROWS][COLS] = {0};
    FILE *input = fopen("founddigits.txt", "r");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (fscanf(input, "%d", &sudoku[i][j]) != 1) {
                printf("Error reading from file.\n");
                fclose(input);
                return 1;
            }
        }
    }

    fclose(input);

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%d ", sudoku[i][j]);
        }
        printf("\n");
    }

    return 0;
}
