Verification folder contains 3 time interval analysis for 3 different drone videos where they are picked to be as different as possible to prevent overfitting down the line.

Video 106 - 2 drones in shot, taking up very small area in the frame
Video 001 - 1 drone from a far
Video 45 - 1 drone at medium range with tractor moving in backround

command:

C:\tmp\prophesee\py3venv\Scripts\python C:\Users\borna\Desktop\Dronovi\scripts\verify_event_labels.py --video "C:\Users\borna\Desktop\Dronovi\Data_raw\Video_V\V_DRONE_106.mp4" --dat "C:\Users\borna\Desktop\Dronovi\Data_new\default_events\V_DRONE_106.dat" --labels "C:\Users\borna\Desktop\Dronovi\Data_new\default_events\V_DRONE_106_LABELS.csv" --output "C:\Users\borna\Desktop\Dronovi\Data_new\verification\V_DRONE_106_20ms" --window_ms 20 --every 30

line:

"C:\Users\borna\Desktop\Dronovi\Data_new\verification\V_DRONE_106_20ms" --window_ms 20 --every 30

Describes that events are grouped in 20ms intervals to form a single image, we are actually saving every 30th one of those since this whole verification folder is only to verify that the boxes that surrounded drones in the original RGB video still encompass the now event based images pretty well.

This has been done with 10ms, 20ms and 40ms to compare the results.

Longer timeframe = more likely to label correctly since there are more events, however more spread out and less precise, also slower reaction
Shorter time = harder to pinpoint drone since very few events coming from it, but fast and precise
