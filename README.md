# abstract-ink

`make_tess.py` create a directory of tessellated images.
To make these into a movie, `cd` into that directory and run
```bash
ffmpeg -f image2 -i frame_%04d.png -c:v libx264 out.mp4
```
