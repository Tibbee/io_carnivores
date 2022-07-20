# Carnivores Export Addon for Blender

This addon is supports exporting Blender models and animations into 3DF / VTL files. 

3DF export simply exports the basic geometry (vertices,faces,active uvlayer) into the 3DF format. Material export is currently not supported.
VTL export basically plays the active scene animation, and captures the vertices of the mesh for every frame in the animation, and writes that to VTL.

These features allow you to do all your modelling/rigging/animating from Blender instead of using the 20+ year old tools that were released with Carnivores.

To create CAR files out of the 3DF and VTL files, you'll have to use either [CMM](https://game3dee.com/cmm/) or [c3dit](https://github.com/carnivores-cpe/c3dit).

## Install

Simply go to Edit/Preferences/Addons, click Install, and select `io_carnivores.py`. You should see the addon in the addon list, click the checkbox in front of its name to enable, and you should be good to go.

## Exporting 3DF

In any Blender document, select the mesh you wish to export as 3DF model, and select File/Export/3DF. Specify the file to be written and any configuration options.

## Exporting VLT

In any Blender document, make sure you have the animation you wish to export active in the scene, then select the mesh (or armature), go to File/Export/VTL. Specify the file to be written and any configuration options.

## Troubleshooting / bugs

This is my first Blender addon, so please be gentle. Known issues:

* Due to the way VTL stores vertices (as signed 16-bit integers, by doing `float * 16`) very small objects or very large ones will
* Likely much more ;)
