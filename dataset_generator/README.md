# Hand Gesture Dataset Generator

This tool captures images of hand gestures using your webcam and saves them as a categorized dataset for computer vision projects.

## How It Works

The program opens your webcam and captures images of hand gestures, labeling them by numbers (0-5). Each number represents a different hand gesture, allowing you to create a labeled dataset for machine learning.

## Requirements

- Python 3.6 or higher
- OpenCV (`pip install opencv-python`)

## Usage

1. Run the program:
   ```
   python hand_gesture_capture.py
   ```

2. When the webcam window opens, you'll see a green rectangle. Position your hand within this rectangle.

3. Press a number key (0-5) to select the gesture category:
   - Hold the key down to capture multiple images continuously
   - When you release the key, one final image will be captured

4. Press ESC to exit the program

## Dataset Structure

Images are saved in the `dataset` folder organized by label:
```
dataset/
  ├── 0/  (images for gesture 0)
  ├── 1/  (images for gesture 1)
  ├── 2/  (images for gesture 2)
  ├── 3/  (images for gesture 3)
  ├── 4/  (images for gesture 4)
  └── 5/  (images for gesture 5)
```

Each image is named with the pattern: `gesture_[label]_[timestamp].jpg`