################################################################################
 # Copyright (C) 2022 Maxim Integrated Products, Inc., All Rights Reserved.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a
 # copy of this software and associated documentation files (the "Software"),
 # to deal in the Software without restriction, including without limitation
 # the rights to use, copy, modify, merge, publish, distribute, sublicense,
 # and/or sell copies of the Software, and to permit persons to whom the
 # Software is furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included
 # in all copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 # IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 # OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 # OTHER DEALINGS IN THE SOFTWARE.
 #
 # Except as contained in this notice, the name of Maxim Integrated
 # Products, Inc. shall not be used except as stated in the Maxim Integrated
 # Products, Inc. Branding Policy.
 #
 # The mere transfer of this software does not imply any licenses
 # of trade secrets, proprietary technology, copyrights, patents,
 # trademarks, maskwork rights, or any other form of intellectual
 # property whatsoever. Maxim Integrated Products, Inc. retains all
 # ownership rights.
 #
 ###############################################################################

"""
Utility functions to generate embeddings and I/O operations
"""

import os
from collections import defaultdict
import numpy as np


from matplotlib.image import imread
from PIL import Image, ExifTags
import torch
import torchvision.transforms.functional as VF


def get_image_rotation(img_path):
    """Reads exif data of the image to get image orientation
    """
    for orientation in ExifTags.TAGS:
        if ExifTags.TAGS[orientation] == 'Orientation':
            break

    image = Image.open(img_path)
    try:
        exif = image._getexif()  # pylint: disable=protected-access
        if exif:
            if orientation in exif:
                return exif[orientation]
    except Exception:  # pylint: disable=broad-except
        print(f'No exif object for {img_path}')

    return 1


def rotate_image(img, img_rot):
    """Rotates image wrt `orientation` value of the exif data
    """
    if img_rot == 1:
        pass
    elif img_rot == 6:
        return np.rot90(img, k=3).copy()
    elif img_rot == 8:
        return np.rot90(img, k=1).copy()
    elif img_rot == 3:
        return np.rot90(img, k=2).copy()
    else:
        print(f'Unknown orientation code: {img_rot}. Image will be used as is.')

    return img


def append_db_file_from_path(folder_path, face_detector, ai85_adapter):
    """Creates embeddings for each image in the given folder and appends to the existing embedding
    dictionary
    """
    subj_id = 0
    subject_list = sorted(os.listdir(folder_path))
    emb_array = np.zeros([1024, 64]).astype(np.uint8)
    recorded_subject = []
    emb_id = 0    
    output_shift = 2 #TODO: Check here for adj. output shift
    summary = {}

    for i in subject_list:
        summary[i] = 0
    for subject in subject_list:
        print(f'Processing subject: {subject}')
        
        subject_path = os.path.join(folder_path, subject)
        if not os.path.isdir(subject_path):
            continue
        if not os.listdir(subject_path):
            subj_id += 1
        for file in os.listdir(subject_path):
            print(f'\tFile: {file}')
            img_path = os.path.join(subject_path, file)
            img_rot = get_image_rotation(img_path)
            img = imread(img_path)
            img = rotate_image(img, img_rot)
            img = img.astype(np.float32)
            
            img = get_face_image(img, face_detector)
            if img is not None:
                img = ((img+1)*128)
                img = (img.squeeze()).detach().cpu().numpy()
                img = img.astype(np.uint8)
                img = img.transpose([1, 2, 0])

                if img.shape == (112, 112, 3):
                    current_embedding = ai85_adapter.get_network_out(img)[0, :]                    
                    current_embedding = current_embedding * 128 * output_shift #convert back to 8 bits after normalization
                    current_embedding = np.clip(current_embedding, -128, 127)  #clamp if embedding > 0.5 or < -0.5 as shift = 2                                      
                    current_embedding[current_embedding < 0] += 256 # Convert negatives to uint
                    current_embedding = np.around(current_embedding).astype(np.uint8).flatten()

                    emb_array[emb_id,:] = current_embedding
                    recorded_subject.append(subject)
                    emb_id += 1
                    summary[subject] += 1

    #np.save('emb_array.npy', emb_array)
    print('Database Summary')
    for key in summary:
        print(f'\t{key}:', f'{summary[key]} images')
    #Format summary for printing image counts per subject



    return emb_array, recorded_subject

def coord_calc(box):
    box[0] = torch.clamp(box[0], min=0) * 168
    box[2] = torch.clamp(box[2], min=0) * 168
    box[1] = torch.clamp(box[1], min=0) * 224
    box[3] = torch.clamp(box[3], min=0) * 224
    
    top = torch.Tensor.int(box[1])
    left = torch.Tensor.int(box[0])
    height = torch.Tensor.int(box[3]-box[1])
    width = torch.Tensor.int(box[2]-box[0])
    return top, left, height, width

def get_face_image(img, face_detector, min_prob=0.25):
    """Detects face on the given image
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = torch.Tensor(img.transpose([2, 0, 1])).to(device).unsqueeze(0)
    
    img = Normalize_Img(img)
    img = VF.resize(img, size=[224, 168])
    with torch.no_grad():
        locs, scores = face_detector.get_network_out(img)
        all_images_boxes, all_images_labels, all_images_scores = \
                face_detector.simulator.model.detect_objects(locs, scores,
                                 min_score=min_prob,
                                 max_overlap=1,
                                 top_k= 1)
        if all_images_labels[0][0] == 1:
            pbox = all_images_boxes[0][0]
            top, left, height, width = coord_calc(pbox)
            img = VF.resized_crop(img= img, top=top, left=left, height=height, width=width, size=[112,112])
            return img

    return None

def create_baseaddr_include_file(baseaddr_h_path):

    data_arr = []
    data_line = []


    baseaddr= ["0x50180000", "0x50184000", "0x50188000", "0x5018c000", "0x50190000", "0x50194000", "0x50198000", "0x5019c000",
        "0x501a0000", "0x501a4000", "0x501a8000", "0x501ac000", "0x501b0000", "0x501b4000", "0x501b8000", "0x501bc000",
        "0x50580000", "0x50584000", "0x50588000", "0x5058c000", "0x50590000", "0x50594000", "0x50598000", "0x5059c000",
        "0x505a0000", "0x505a4000", "0x505a8000", "0x505ac000", "0x505b0000", "0x505b4000", "0x505b8000", "0x505bc000",
        "0x50980000", "0x50984000", "0x50988000", "0x5098c000", "0x50990000", "0x50994000", "0x50998000", "0x5099c000",
        "0x509a0000", "0x509a4000", "0x509a8000", "0x509ac000", "0x509b0000", "0x509b4000", "0x509b8000", "0x509bc000",
        "0x50d80000", "0x50d84000", "0x50d88000", "0x50d8c000", "0x50d90000", "0x50d94000", "0x50d98000", "0x50d9c000",
        "0x50da0000", "0x50da4000", "0x50da8000", "0x50dac000", "0x50db0000", "0x50db4000", "0x50db8000", "0x50dbc000"]
    
    if baseaddr_h_path is not None:
        with open(baseaddr_h_path, 'w') as h_file:
            h_file.write('#define BASEADDR { \\\n  ')
            for base in baseaddr:
                data_line.append(base)
                if (len(data_line) % 8) == 0:
                    data_arr.append(', '.join(data_line))
                    data_line.clear()

            data_arr.append(', '.join(data_line))
            data = ', \\\n  '.join(data_arr)
            h_file.write(data)
            h_file.write(' \\\n}')
            h_file.write('\n')

    return baseaddr

def create_weights_include_file(emb_array, weights_h_path, baseaddr):
    """
    Create weights_3.h file from embeddings
    """
    Embedding_dimension = 64
    extension = os.path.splitext(weights_h_path)[1]

    length = "0x00000101"
    data_arr = []
    data_line = []
    

    if extension == '.h':
        with open(weights_h_path, 'w') as h_file:
            four_byte = []
            h_file.write('#define KERNELS_3 { \\\n  ')
            for dim in range(Embedding_dimension):
                init_proccessor = False
                for i in range(emb_array.shape[0] + 4): # nearest %9 == 0 for 1024 is 1027, it can be kept in 1028 bytes TODO: Change this from Hardcoded
                    reindex = i + 8 - 2*(i%9)
                    if reindex < 1024: # Total emb count is 1024, last index 1023
                        single_byte = str(format(emb_array[reindex][dim], 'x')) #Relocate emb for cnn kernel
                    else:
                        single_byte = str(format(0, 'x'))
                    if len(single_byte) == 1:
                        single_byte = '0' + single_byte
                    four_byte.append(single_byte)

                    if (i + 1) % 4 == 0:
                        if not init_proccessor:
                            data_line.append(baseaddr[dim])
                            if (len(data_line) % 8) == 0:
                                data_arr.append(', '.join(data_line))
                                data_line.clear()
                            data_line.append(length)
                            if (len(data_line) % 8) == 0:
                                data_arr.append(', '.join(data_line))
                                data_line.clear()
                            init_proccessor = True
                        data_32 = '0x'+''.join(four_byte)

                        data_line.append(data_32)
                        four_byte.clear()
                        if (len(data_line) % 8) == 0:
                            data_arr.append(', '.join(data_line))
                            data_line.clear()

            data_arr.append(', '.join(data_line))
            data = ', \\\n  '.join(data_arr)
            h_file.write(data)
            h_file.write('0x00000000') #TODO: CHECK THE SOURCE OF THE LAST 0x00000000, PS: Might be due to addr matching while loading weights at c side
            h_file.write(' \\\n}')
            h_file.write('\n')

    elif extension == '.bin':
        with open(weights_h_path, 'wb') as h_file:
            four_byte = 0
            data_arr = bytearray(np.uint8([data_arr]))
            for dim in range(Embedding_dimension):
                for i in range(emb_array.shape[0] + 4): # nearest %9 == 0 for 1024 is 1027, it can be kept in 1028 bytes TODO: Change this from Hardcoded
                    reindex = i + 8 - 2*(i%9)
                    if reindex < 1024: # Total emb count is 1024, last index 1023
                        single_byte = int(emb_array[reindex][dim])  #Relocate emb for cnn kernel                        
                    else:
                        single_byte = 0
                    four_byte = four_byte << 8 | single_byte
                    if (i + 1) % 4 == 0:
                        data_arr.extend(four_byte.to_bytes(4, 'little', signed=False))
                        four_byte = 0
            h_file.write(data_arr)
            

def create_embeddings_include_file(recorded_subjects, embeddings_h_path):
    data_arr = []
    data_line = []

    with open(embeddings_h_path, 'w') as h_file:        
        h_file.write('#define DEFAULT_EMBS_NUM ' + str(len(recorded_subjects)) + ' \n')
        h_file.write('\n')
        h_file.write('#define DEFAULT_NAMES { \\\n ')

        for subject in recorded_subjects:
            if len(subject) > 6: #TODO: 6 is the max name length for now
                subject = subject[:6]
            data_line.append('"' + subject + '"')
            if (len(data_line) % 15) == 0:
                data_arr.append(', '.join(data_line))
                data_line.clear()

        data_arr.append(', '.join(data_line))
        data = ', \\\n '.join(data_arr)
        h_file.write(data)
        h_file.write(' \\\n}')
        h_file.write('\n')

def Normalize_Img(img):
    return img.sub(128).clamp(min=-128, max=127).div(128.)
