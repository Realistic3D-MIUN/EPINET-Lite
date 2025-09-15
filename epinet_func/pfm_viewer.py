import numpy as np
import os
import matplotlib.pyplot as plt

def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data


dir_test_LFimages = [
        'stratified/backgammon', 'stratified/dots', 'stratified/pyramids', 'stratified/stripes',
        'training/boxes', 'training/cotton', 'training/dino', 'training/sideboard',
        'test/bedroom', 'test/bicycle', 'test/herbs', 'test/origami',
    ]
dir_test_LFimages = [
        'backgammon', 'dots', 'pyramids', 'stripes',
        'boxes', 'cotton', 'dino', 'sideboard',
        'bedroom', 'bicycle', 'herbs', 'origami',
    ]

image_h = 512
image_w = 512
indx = 0
test_labels = np.zeros((len(dir_test_LFimages), 1, 512, 512))
data_path = "C:/Users/alihas/MyData/AliHassan/Python/hci_submission/algo_results/EPINET-Lite/disp_maps/"
for image_path in dir_test_LFimages:
    if os.path.exists(os.path.join(data_path + image_path + '.pfm')):
        label = read_pfm(os.path.join(data_path + image_path + '.pfm'))
        test_labels[indx, 0] = label
        # Normalize label for saving as 8-bit PNG (optional, depending on your needs)
        # Normalize disparity map to 0–1
        label_normalized = (label - np.min(label)) / (np.max(label) - np.min(label) + 1e-8)

        # Save with a Matplotlib colormap (e.g., 'viridis', 'plasma', 'inferno', etc.)
        save_path = os.path.join(data_path, image_path + '.png')
        plt.imsave(save_path, label_normalized, cmap='viridis')  # or 'plasma', 'inferno', etc.
    indx = indx + 1