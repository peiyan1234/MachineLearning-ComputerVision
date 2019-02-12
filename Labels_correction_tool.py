import json
import os
import glob
import numpy as np
from PIL import Image

"""
This tool is created by Alvin Pei Yan, Li, whose email addresses are Alvin.Li@acer.com and d05548014@ntu.edu.tw.
Please contact to him, the author, if any problems or requirments has been found or asked.
2019.02.12 for the 2nd version.
"""

"Global Variables"

global dir_labels
global dir_savingfiles

"Function Definitions"

def main():
    """
    This is the main function handling all sub-functions together.
    """
    print('Start.\n')
    goback_dir = os.getcwd()

    global_IDtable = get_global_table()
    correct_local_labelfiles(global_IDtable)

    os.chdir(goback_dir)
    print('\nDone.')

def get_global_table():
    """
    get the correction table for correcting labels and return it as a global_IDtable
    the name: label_to_id
    the suffix: .json
    the format: hash table
    """
    try:
        with open('label_to_id.json', encoding='utf-8-sig') as json_file:
            print('label_to_id.json has been read.')

            global_IDtable = json.load(json_file)
    except IOError:
            print('An error occured trying to read {}.'.format('label_to_id.json'))

    return global_IDtable

def correct_local_labelfiles(global_IDtable):
    """
    do the label correction for all the label files in labels/ and store the corrected files in the subdir correct/
    """
    def correct_single_npyfile(local_label_arr, local_label_filename, local_label_table, global_IDtable):
        """
        do the correction mapping for each local .npy label file
        """

        corrected_label_arr = np.zeros(local_label_arr.shape)

        for key in list(local_label_table.keys()):
            corrected_label_arr[local_label_arr == int(key)] = global_IDtable[local_label_table[key]]

        new_label_filename = local_label_filename + '_n'

        savingdir = dir_savingfiles + '/' + new_label_filename
        np.save(savingdir, corrected_label_arr)

        print('{} has been saved.'.format(new_label_filename))

        return corrected_label_arr

    def generate_corrected_seg(corrected_label_arr, local_label_filename, global_IDtable):
        """
        generate a corrected segamenation mask
        """

        frameshape = list(corrected_label_arr.shape)
        colored_new_seg_im = np.zeros((frameshape[0], frameshape[1], 3), dtype = np.uint8) # BRK: dtype:uint8

        stacked_for_boolean_masks = np.stack((corrected_label_arr,corrected_label_arr,corrected_label_arr), axis = 2)

        ID_dict = get_ID_dict(global_IDtable)
        for tissue_name in list(ID_dict.values()):
            tissueID = int(global_IDtable[tissue_name])
            tissue_mask = get_tissue_mask(tissueID, stacked_for_boolean_masks)
            colored_new_seg_im[tissue_mask] = draw_color(tissue_name)

        name_of_new_seg_im = local_label_filename + '_n_seg.jpg'

        new_seg_im = Image.fromarray(np.uint8(colored_new_seg_im))

        savingdir = dir_savingfiles + '/' + name_of_new_seg_im
        new_seg_im.save(savingdir)

        print('{} has been saved.'.format(name_of_new_seg_im))

        return name_of_new_seg_im

    def get_ID_dict(global_IDtable):

        ID_dict = dict(zip(range(len(list(global_IDtable.keys()))),list(global_IDtable.keys())))

        return ID_dict

    def get_tissue_mask(tissueID, stacked_for_boolean_masks):

        tissue_mask = (stacked_for_boolean_masks == [tissueID, tissueID, tissueID]).all(axis = 2)

        return tissue_mask

    def draw_color(tissue_name):

        RGB_dict = { 'Black' : [0, 0, 0],
                     'White' : [255, 255, 255],
                     'Medium Gray' : [128, 128, 128],
                     'Aqua' : [0, 128, 128],
                     'Navy Blue' : [0, 0, 128],
                     'Green' : [0, 255, 0],
                     'Orange' : [255, 165, 0],
                     'Yellow' : [255, 255, 0],
                     'Maroon' : [128, 0, 0]
                    }

        tissue_color = { 'Background' : 'Black',
                         'Peritoneum' : 'Maroon',
                         'Ovary' : 'Medium Gray',
                         'Uterus' : 'White',
                         'Fallopian_Tube' : 'Navy Blue',
                         'Ligament' : 'Green',
                         'Ureter' : 'Orange',
                         'Artery' : 'Yellow',
                         'Scapel' : 'Aqua',
                       }

        RGB_color = RGB_dict[tissue_color[tissue_name]]

        return RGB_color

    def generate_overlayimage(name_of_new_seg_im, local_label_filename):
        """
        generate an image overlaying the original image with the corresponding mask
        """
        name_of_original_image = local_label_filename + '.jpg'

        open_dir_original = dir_labels + '\\' + name_of_original_image
        original_im = Image.open(open_dir_original)

        open_dir_new = dir_savingfiles + '\\' + name_of_new_seg_im
        new_seg_im = Image.open(open_dir_new)
        original_im = original_im
        new_seg_im = new_seg_im

        overlay_image = Image.blend(original_im, new_seg_im, 0.4)   # 0.4 is the transparent rate.

        name_of_overlay_image = local_label_filename + '_n_overlay.jpg'
        savingdir = dir_savingfiles + '/' + name_of_overlay_image
        overlay_image.save(savingdir)

        print('{} has been saved.'.format(name_of_overlay_image))

    """
    realize the functionality of 'correct_local_labelfiles'
    """
    os.chdir(dir_labels)
    os.mkdir(dir_savingfiles)

    for filename in glob.glob('*.json'):
        print(filename + 'has been found.')
        local_label_filename = filename.replace(".json","")
        local_label_arr = np.load(local_label_filename + '.npy')

        try:
            with open(filename, encoding='utf-8-sig') as json_file:
                local_label_table = json.load(json_file)
        except IOError:
            print('An error occured trying to read {}.'.format(filename))

        corrected_label_arr = correct_single_npyfile(local_label_arr, local_label_filename, local_label_table, global_IDtable)
        name_of_new_seg_im = generate_corrected_seg(corrected_label_arr, local_label_filename, global_IDtable)
        generate_overlayimage(name_of_new_seg_im, local_label_filename)

""" Execution """

dir_labels = input('Input the path of labels directory: ')
dir_savingfiles = input('Input the path of the directory created for saveing files: ')

import time
if __name__ == '__main__':
    start = time. time()

    main()

    end = time. time()

    duration = end - start

    print('\nThis code runs so fast that only spends {} in second.'.format(duration))
