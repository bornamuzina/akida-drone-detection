RGB videos mp4 connected with respective labels .mat -> from Data_raw

WARNING: do not use these as simulator input. The boxes are drawn into
the pixels, so the event simulator generates events along the rectangle
edges -- exactly where the label says the drone is. A model trained on
that learns to find the rectangle, scores well on this data, and finds
nothing on real footage. Use Data_raw\Video_V instead.

Produced by src\data\MP4+.mat_TO_RGB.py. For visual checking only.
