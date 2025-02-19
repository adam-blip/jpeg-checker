import argparse
import sys
from PIL import Image, ImageFile, ImageStat
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import mimetypes
from io import BytesIO

# Disable truncation handling in Pillow
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Known JPEG markers and whether they have a length field
MARKER_DICT = {
    0xC0: ('SOF0', True), 0xC1: ('SOF1', True), 0xC2: ('SOF2', True), 0xC3: ('SOF3', True),
    0xC5: ('SOF5', True), 0xC6: ('SOF6', True), 0xC7: ('SOF7', True), 0xC8: ('JPG', True),
    0xC9: ('SOF9', True), 0xCA: ('SOF10', True), 0xCB: ('SOF11', True), 0xCD: ('SOF13', True),
    0xCE: ('SOF14', True), 0xCF: ('SOF15', True), 0xC4: ('DHT', True), 0xCC: ('DAC', True),
    0xD8: ('SOI', False), 0xD9: ('EOI', False), 0xDA: ('SOS', True), 0xDB: ('DQT', True),
    0xDC: ('DNL', True), 0xDD: ('DRI', True), 0xDE: ('DHP', True), 0xDF: ('EXP', True),
    0xE0: ('APP0', True), 0xE1: ('APP1', True), 0xE2: ('APP2', True), 0xE3: ('APP3', True),
    0xE4: ('APP4', True), 0xE5: ('APP5', True), 0xE6: ('APP6', True), 0xE7: ('APP7', True),
    0xE8: ('APP8', True), 0xE9: ('APP9', True), 0xEA: ('APP10', True), 0xEB: ('APP11', True),
    0xEC: ('APP12', True), 0xED: ('APP13', True), 0xEE: ('APP14', True), 0xEF: ('APP15', True),
    0xFE: ('COM', True),
}

def is_valid_jpeg(filename):
    """Check if the file is a valid JPEG based on its MIME type."""
    mime, _ = mimetypes.guess_type(filename)
    return mime == 'image/jpeg'

def check_jpeg(filename):
    """Check a JPEG file for structural errors, decoding issues, and compare with PNG conversion."""
    errors = []
    warnings = []

    # Initialize logging
    logging.info(f"Checking file: {filename}")

    # Check if the file is a valid JPEG
    if not is_valid_jpeg(filename):
        errors.append(f"File '{filename}' is not a valid JPEG")
        logging.error("File is not a valid JPEG.")
        return errors, warnings

    try:
        with open(filename, 'rb') as f:
            data = f.read()
    except Exception as e:
        errors.append(f"Failed to open file: {str(e)}")
        logging.error(f"Failed to open file: {str(e)}")
        return errors, warnings

    # Basic file checks
    if len(data) < 2:
        errors.append("File too small to be a JPEG")
        logging.error("File too small to be a JPEG.")
        return errors, warnings

    # Check SOI marker
    if data[:2] != b'\xff\xd8':
        errors.append("Missing SOI (Start Of Image) marker")
        logging.error("Missing SOI (Start Of Image) marker.")
    else:
        logging.info("SOI marker found.")

    # Check EOI marker
    if data[-2:] != b'\xff\xd9':
        errors.append("Missing EOI (End Of Image) marker")
        logging.error("Missing EOI (End Of Image) marker.")
    else:
        logging.info("EOI marker found.")

    # Check for extra data after EOI
    if data[-2:] == b'\xff\xd9' and len(data) > 2 and len(data) != data.rfind(b'\xff\xd9') + 2:
        warnings.append("Extra data detected after EOI marker")
        logging.warning("Extra data detected after EOI marker.")

    pos = 2  # Current position in file
    sos_found = False
    eoi_found = False
    dqt_found = False
    sof_found = False
    sos_data_start = None

    # Parse markers
    while pos < len(data):
        # Find next marker
        if data[pos] != 0xFF:
            errors.append(f"Expected marker at position {pos}, found 0x{data[pos]:02X}")
            logging.error(f"Expected marker at position {pos}, found 0x{data[pos]:02X}.")
            break

        # Skip any padding FF bytes
        while data[pos] == 0xFF:
            pos += 1
            if pos >= len(data):
                errors.append("Reached end of file while searching for marker")
                logging.error("Reached end of file while searching for marker.")
                return errors, warnings

        marker = data[pos]
        pos += 1
        marker_name, has_length = MARKER_DICT.get(marker, (f'Unknown (0x{marker:02X})', False))

        logging.info(f"Found marker: {marker_name} (0x{marker:02X})")

        if marker_name == 'SOS':
            sos_found = True
            sos_data_start = pos  # Track start of SOS data
            logging.info("SOS (Start Of Scan) marker found.")

        if has_length:
            if pos + 1 >= len(data):
                errors.append(f"Truncated length field for marker {marker_name}")
                logging.error(f"Truncated length field for marker {marker_name}.")
                break

            length = (data[pos] << 8) + data[pos + 1]
            pos += 2

            if pos + length - 2 > len(data):
                errors.append(f"Length field for marker {marker_name} exceeds file size")
                logging.error(f"Length field for marker {marker_name} exceeds file size.")
                break

            pos += length - 2

            if marker_name.startswith('DQT'):
                dqt_found = True
                logging.info("Quantization table (DQT) found.")
            elif marker_name.startswith('SOF'):
                sof_found = True
                logging.info(f"Frame header ({marker_name}) found.")
        else:
            if marker == 0xD9:  # EOI marker
                eoi_found = True
                logging.info("EOI (End Of Image) marker found.")
                break  # No more data after EOI

    if not sof_found:
        errors.append("Missing SOF (Start Of Frame) marker")
        logging.error("Missing SOF (Start Of Frame) marker.")
    if not dqt_found:
        warnings.append("No quantization table (DQT) found")
        logging.warning("No quantization table (DQT) found.")
    if not sos_found:
        errors.append("Missing SOS (Start Of Scan) marker")
        logging.error("Missing SOS (Start Of Scan) marker.")
    if not eoi_found:
        errors.append("Missing EOI (End Of Image) marker")
        logging.error("Missing EOI (End Of Image) marker.")

    # Attempt to decode and convert to PNG in-memory
    try:
        logging.info("Attempting to open image with Pillow.")
        with Image.open(filename) as img:
            if img is None:
                raise ValueError("Image object is None")
            logging.info("Image object created successfully.")

            # Verify image integrity
            img.verify()
            logging.info("Image verified successfully.")

            # Load image data
            img = Image.open(filename)
            img.load()
            logging.info("Image data loaded successfully.")

            # Check resolution
            width, height = img.size
            logging.info(f"Image resolution: {width}x{height}")

            # Check color mode
            color_mode = img.mode
            logging.info(f"Image color mode: {color_mode}")

            # Check compression quality (estimate)
            stat = ImageStat.Stat(img)
            if stat.extrema:
                logging.info(f"Image compression quality estimate: {stat.extrema}")

            # Convert to RGB and simulate PNG conversion
            img_rgb = img.convert('RGB')
            logging.info("Image conversion to RGB successful.")

            # Save as PNG in-memory to compare
            png_buffer = BytesIO()
            img_rgb.save(png_buffer, format='PNG')
            png_buffer.seek(0)
            png_img = Image.open(png_buffer)

            # Compare resolution and color mode with PNG
            png_width, png_height = png_img.size
            png_color_mode = png_img.mode
            if (width, height) != (png_width, png_height):
                warnings.append("Resolution mismatch between JPEG and PNG conversion.")
                logging.warning("Resolution mismatch between JPEG and PNG conversion.")
            if color_mode != png_color_mode:
                warnings.append("Color mode mismatch between JPEG and PNG conversion.")
                logging.warning("Color mode mismatch between JPEG and PNG conversion.")

            logging.info("Image comparison with PNG conversion successful.")

    except Exception as e:
        errors.append(f"Image decoding failed: {str(e)}")
        logging.error(f"Image decoding failed: {str(e)}")

    return errors, warnings

def main():
    """Main function to process command-line arguments and check JPEG files."""
    parser = argparse.ArgumentParser(description='Check JPEG files for errors')
    parser.add_argument('filenames', nargs='+', help='JPEG files to check')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    def process_file(filename):
        """Process a single file and return its results."""
        errors, warnings = check_jpeg(filename)
        result = {
            'filename': filename,
            'errors': errors,
            'warnings': warnings
        }
        return result

    results = []
    if args.no_parallel:
        for filename in args.filenames:
            result = process_file(filename)
            results.append(result)
    else:
        max_workers = min(32, os.cpu_count() + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file, filename): filename for filename in args.filenames}
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

    all_errors = []
    all_warnings = []
    for result in results:
        filename = result['filename']
        errors = result['errors']
        warnings = result['warnings']
        print(f"\nChecking file: {filename}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"• {warning}")
            all_warnings.extend(warnings)
        if errors:
            print("\nErrors found:")
            for error in errors:
                print(f"• {error}")
            all_errors.extend(errors)
        else:
            print("No errors detected.")

    if all_errors:
        sys.exit(1)
    elif all_warnings:
        print("\nNo critical errors detected, but some warnings were found.")
        sys.exit(0)
    else:
        print("\nAll files appear structurally valid and decodable.")
        sys.exit(0)

if __name__ == "__main__":
    main()
