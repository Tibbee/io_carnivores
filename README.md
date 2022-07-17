# Carnivores Export Addon for Blender

This is a first stab at creating an addon to support modelling/rigging/animating classic Carnivores models in Blender.

NOTE: This addon has only been tested with Blender 2.92.0

## Install

Simply go to Edit/Preferences/Addons, click Install, and select `io_carnivores.py`. You should see the addon in the addon list, click the checkbox before its name to enable, and you should be good to go.

## Exporting 3DF

Simply open any Blender file, select the mesh you wish to export as 3DF model, and select File/Export/3DF. Specify name/location of output file, and you should have a 3DF file usable in CMM / C3dit, etc.

## Exporting VLT

In any Blender project, select an active animation, then select the mesh, go to File/Export/VTL, choose name/location of output file, and you should be good to go.

## Troubleshooting / bugs

This is my first Blender addon, so please be gentle. Known issues:

* Due to the VTL format, very small objects will show artifacts due to limited X/Y/Z value range.
* Using a model without uv layer gives script errors
* Selecting something else then a mesh will cause script errors
* Likely much more ;)

Feel free to let me know any additional issues you run into, or any features you would like to see added.
