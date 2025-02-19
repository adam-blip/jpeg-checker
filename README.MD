# JPEG File Checker

This script checks JPEG files for structural errors, decoding issues, and compares them with their PNG conversion. It provides detailed logging and progress tracking, ensuring that the images are valid and correctly formatted.

## Features

- Verifies the structural integrity of JPEG files.
- Checks for decoding issues and logs detailed progress.
- Compares resolution, color mode, and compression with PNG conversion.
- Supports parallel processing for multiple files.
- Provides verbose logging for detailed insights.

## Requirements

- Python 3.x
- Pillow library (`PIL`)

## Installation

1. Clone the repository:
2. Install the required dependencies:

pip install pillow

## Usage

python check_jpeg.py [filenames] [--no-parallel] [--verbose]
filenames: List of JPEG files to check.
--no-parallel: Disable parallel processing.
--verbose: Enable verbose logging.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments
Thanks to the Pillow library for image processing capabilities.