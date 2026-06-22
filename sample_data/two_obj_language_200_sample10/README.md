# Two-Object Language Sample10

Small real-data sample from the Piper SmolVLA blue/green two-object collection.

## Contents

| Item | Value |
| --- | --- |
| Episodes | 10 |
| Frames | 2126 |
| FPS | 20 |
| Tasks | blue object / green object |
| Cameras | global RGB + wrist RGB |
| Video codec | H.264 |

## Structure

```text
data/chunk-000/file-000.parquet
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
meta/two_object_language_schedule.json
meta/two_object_language_episode_metadata/episode_*.json
videos/observation.images.global_rgb/chunk-000/file-000.mp4
videos/observation.images.wrist_rgb/chunk-000/file-000.mp4
```

This is a compact GitHub sample, not the full training dataset.
