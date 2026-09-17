When: after the first verification run returned 305 label entries that were all None, so no boxes were drawn on any frame.

Why: the .mat files don't contain a plain array. They hold a MATLAB object — a groundTruth instance from Video Labeler, identifiable by the MatlabOpaque wrapper and a 177 KB **function_workspace** blob. Objects are serialized in an undocumented format, and the class definition that reconstructs them ships inside MATLAB's Computer Vision Toolbox. scipy could see the object was there but had no way to unpack it.

What was done: installed MATLAB R2026a (student licence) with Computer Vision Toolbox and its Image Processing dependency, loaded gTruth, and looped over gTruth.LabelData writing frame, x, y, w, h to CSV — flattening two boxes per frame into one row each, converting timestamps to 0-based frame indices, and dropping the three empty classes.

Result: 608 rows, 304 frames, readable by anything. MATLAB isn't needed again except to export the remaining clips' labels.
