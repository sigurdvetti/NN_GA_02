# Snake Reinforcement Learning

Code for training a Deep Reinforcement Learning agent to play the game of Snake.
The agent takes 2 frames of the game as input (image) and predicts the action values for
the next action to take.
***

# Running Graded Assignment 02

Run training.py to train the model.
Run game_visualization.py to generate .mp4 files with examples of models playing the game.
(fill the "iterations" list with the iteration numbers of the models you want to test)

# Additional Dependencies

The game_visualization.py requires you to use ffmpeg
https://www.gyan.dev/ffmpeg/builds/
- Download "ffmpeg-2024-11-25-git-04ce01df0b-full_build.7z"
- Extract the file
- Add the path to the extracted folder's /bin to system PATH
- Restart terminal or do $env:Path += PATH_TO_BIN

If there is any confusion regarding the code, look for docstrings and comments. It should be thoroughly documented.