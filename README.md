
This is an implementation of Conway's Game of Life in pure python based on this implementation of the HashLife algorithm: https://johnhw.github.io/hashlife/index.md.html 

This implementation of HashLife sets the game board to 2^20 x 2^20 cells and displays the center of the board in the terminal. The exact area displayed is determined by the following formula:
```python
int(log2(min(terminal_size.columns//3, terminal_size.lines-1)))
```

To run the script simply pass a [plaintext](https://conwaylife.com/wiki/Plaintext) file as the argument for the `text_life_points`
function.