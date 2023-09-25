# tracksync: Comparing your dashcam to their dashcam

Tracksync is a simple wrapper around ffmpeg to autoscale video A to a reference
video B by matching timestamps at various milestones. I use this script to
compare my racing dashcam against my friends to determine how closely our lines
match, despite differing overall lap times.

Tracksync scales the videos such that the cameras reach milestones (e.g., turn
apexes) at the same time. Then, the videos are vertically stacked for comparison.
Over time, I may extend the script a bit to include apex speeds and other
improvements.

Tracksync accepts a CSV of the format:

```
milestones, video1,, video2,, video3,,
start, 0.0, 100.0, 0.1, 100.2, 0.25, 103.5
m1, 12.3, 78.9, 11.2, 77.8, 10.3, 98.7
m2, ...
```

The first column is the milestone name. The second column indicates the basename of
the first video. The fourth column is the basename of the second video, and so on.
After the header row, each consecutive row indicates the milestone name and the
timestamp (in seconds) and speed (in mph) for each video.

Usage: `./tracksync.py timestamps.csv`

The script outputs videos in the form video1_vs_video2.mp4, where video1 is scaled
to reference video2.
