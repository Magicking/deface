#!/bin/bash

# Script that tests black-box tests deface with examples.
# Before running, install deface and cd to the repo root

set -e


# Show help, checking if the main script runs without errors
deface --help

# Create a temporary directory for outputs
tmpdir=$(mktemp -d -t deface-XXXXXXXXXX)

# Test deface with the example image, write output to temporary directory
deface ./examples/city.jpg -o ${tmpdir}/city_anonymized.jpg

# Keep faces in a zone visible
deface ./examples/city.jpg --keep-zone 0,0,0.5,1 -o ${tmpdir}/city_keep_zone.jpg

# Select faces to keep interactively (answer "0" on stdin), then reuse the saved selection
echo 0 | deface ./examples/city.jpg --select-faces -o ${tmpdir}/city_select.jpg
test -f ${tmpdir}/city_select_faces/face_00.png
deface ./examples/city.jpg --keep-face ${tmpdir}/city_select_faces/keep.npz -o ${tmpdir}/city_keep_npz.jpg
cmp ${tmpdir}/city_select.jpg ${tmpdir}/city_keep_npz.jpg

# Keep the person shown in a reference picture
deface ./examples/city.jpg --keep-face ${tmpdir}/city_select_faces/face_00.png -o ${tmpdir}/city_keep_face.jpg

# TODO: Add more tests (video I/O, different EPs)
