 Emoji Face Expression Overlay

A fun real-time Python project that detects facial expressions using MediaPipe's Face Detection and Face Mesh modules, and overlays appropriate emojis on the user's face based on their mouth openness (MAR - Mouth Aspect Ratio).

![demo](demo.gif)

## 🚀 Features

- Real-time webcam input
- Face detection and facial landmark tracking with MediaPipe
- Mouth Aspect Ratio (MAR) calculation for expression detection
- Emoji overlay with transparency support
- Smooth landmark tracking to reduce jitter
- Automatically categorizes expressions into:
  - 😀 Smiling
  - 😐 Neutral
  - 😮 Surprised

## 🧠 Expression Logic

The MAR (Mouth Aspect Ratio) is calculated using vertical and horizontal distances between facial landmarks. Based on the MAR:

| MAR Range      | Expression |
|----------------|------------|
| > 0.55         | Surprised 😮 |
| 0.42 - 0.55    | Smiling 😀 |
| < 0.42         | Neutral 😐 |

## 📂 Folder Structure

