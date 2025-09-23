# Elegant Snake

A polished Snake game built with [pygame](https://www.pygame.org/) featuring a spacious arena, smooth controls, and a persistent user high-score system.

## Features

- **User profiles:** choose a player name on launch or press `U` to switch users at any time.
- **Persistent high scores:** each player's personal best is stored in `highscores.json` alongside the game, so records survive between play sessions.
- **Responsive controls:** fluid movement on a 30×24 grid with tasteful gradients and HUD overlays.

## Getting started

1. Ensure Python 3.9+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the game:
   ```bash
   python main.py
   ```

A prompt will appear to capture your player name before the game begins.

## Distributing the game

To share the game so that someone can unzip it and play immediately, bundle the code together with the Python runtime using a tool such as [PyInstaller](https://pyinstaller.org/):

```bash
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --windowed --name ElegantSnake main.py
```

The generated bundle will be placed in `dist/ElegantSnake/`. Zip that folder and send it to your players—they can extract it and run the `ElegantSnake` executable without needing Python installed.

> **Tip:** if you plan to distribute to other operating systems, run PyInstaller on that target platform to ensure compatible binaries.

## Saving location

High scores are stored in `highscores.json` next to `main.py`. Delete this file to reset all records.

## License

This project is provided as-is for demonstration purposes.
