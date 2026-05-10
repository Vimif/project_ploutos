import os

filepath = ".github/workflows/tests.yml"
with open(filepath, "r") as f:
    content = f.read()

# Replace torch version requirement to something stable if that is the issue? Wait no, let's see. The error happens during model initialization.
