# CAD Based pose estimation for bin-picking
Tim Bossuyt

## To-Do
* Define picking poses
* Use multiple AruCo markers instead of 1 for transformation estimation
* Update pose_estimation docs

## Notes
* Changed CAD-model to only upper half to remove dubious surface normals

## Coordinate systems
### Camera
Defined by camera
### World
Defined by center of one AruCo marker (X,Y) and normal of the dominant plane (Z)
### Cobot
Defined by cobot