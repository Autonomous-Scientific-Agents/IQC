# XYZ Molecule Viewer

A simple, browser-based molecular structure viewer for XYZ files. This viewer allows you to visualize molecular structures and trajectories directly in your web browser without the need for installation or server-side processing.

## Features

- **Single File Viewing**: Load and view individual XYZ files
- **Directory/Trajectory Support**: Load multiple XYZ files from a directory
- **Interactive Navigation**: 
  - Browse through structures using arrow keys (↑/↓)
  - Click to select files from the sidebar
- **Molecular Visualization**:
  - Ball and stick representation
  - Interactive 3D view with rotation and zoom
  - White background for clarity
- **File Information**: Displays the comment line (second line) from XYZ files
- **Responsive Design**: Adapts to different screen sizes

## Usage

1. Open `molecule_viewer.html` in a web browser
2. Click the file input button and select either:
   - A single XYZ file
   - A directory containing multiple XYZ files
3. Navigate through files using:
   - Up/Down arrow keys
   - Mouse clicks on file names in the sidebar

## File Format

The viewer expects XYZ files in the standard format:
```
number_of_atoms
comment_line
atom_symbol x y z
...
```

Coordinates should be in Angstroms.

## Browser Compatibility

The viewer works in modern web browsers that support:
- ES6+ JavaScript
- WebGL for 3D rendering
- File System Access API for directory selection
## Technical Details

The viewer is built using:
- Pure HTML, CSS, and JavaScript
- NGL Viewer for molecular visualization


## Credits and Citation

This viewer is built on top of the NGL Viewer library. Please check the [NGL Viewer Documentation](http://nglviewer.org/ngl/api/) for more information. 

NGL Viewer is licensed under the MIT license. For more information, visit:
- [NGL Viewer GitHub Repository](https://github.com/nglviewer/ngl)
- [NGL Viewer Documentation](http://nglviewer.org/ngl/api/)


## License

This project is licensed under the MIT License - see the LICENSE file for details. 
