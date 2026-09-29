import numpy as np
import nibabel as nib
import sys

def map_image(img, out_affine, out_shape, ras2ras=np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
              order=1):
    from scipy.ndimage import affine_transform
    from numpy.linalg import inv

    vox2vox = inv(out_affine) @ ras2ras @ img.affine

    new_data = affine_transform(img.get_fdata(), inv(vox2vox), output_shape=out_shape, order=order)
    return new_data

def getscale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):

    data = np.clip(data, a_min=0, a_max=np.max(data))

    src_min = np.min(data)
    src_max = np.max(data)

    if src_min < 0.0:
        sys.exit('ERROR: Min value in input is below 0.0!')

    print("Input:    min: " + format(src_min) + "  max: " + format(src_max))

    if f_low == 0.0 and f_high == 1.0:
        return src_min, 1.0

    nz = (np.abs(data) >= 1e-15).sum()
    voxnum = data.shape[0] * data.shape[1] * data.shape[2]

    histosize = 1000
    bin_size = (src_max - src_min) / histosize
    hist, bin_edges = np.histogram(data, histosize)

    cs = np.concatenate(([0], np.cumsum(hist)))

    nth = int(f_low * voxnum)
    idx = np.where(cs < nth)

    if len(idx[0]) > 0:
        idx = idx[0][-1] + 1

    else:
        idx = 0

    src_min = idx * bin_size + src_min

    nth = voxnum - int((1.0 - f_high) * nz)
    idx = np.where(cs >= nth)

    if len(idx[0]) > 0:
        idx = idx[0][0] - 2

    else:
        print('ERROR: rescale upper bound not found')

    src_max = idx * bin_size + src_min

    if src_min == src_max:
        scale = 1.0

    else:
        scale = (dst_max - dst_min) / (src_max - src_min)

    print("rescale:  min: " + format(src_min) + "  max: " + format(src_max) + "  scale: " + format(scale))

    return src_min, scale

def scalecrop(data, dst_min, dst_max, src_min, scale):
    data_new = dst_min + scale * (data - src_min)

    data_new = np.clip(data_new, dst_min, dst_max)
    print("Output:   min: " + format(data_new.min()) + "  max: " + format(data_new.max()))

    return data_new

def rescale(data, dst_min, dst_max, f_low=0.0, f_high=0.999):
    src_min, scale = getscale(data, dst_min, dst_max, f_low, f_high)
    data_new = scalecrop(data, dst_min, dst_max, src_min, scale)
    return data_new

def conform(img, order=1, rescale=False, imagetype='image'):
    from nibabel.freesurfer.mghformat import MGHHeader

    cwidth = 256
    csize = 1
    h1 = MGHHeader.from_header(img.header)

    h1.set_data_shape([cwidth, cwidth, cwidth, 1])
    h1.set_zooms([csize, csize, csize])
    h1['Mdc'] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    h1['fov'] = cwidth
    h1['Pxyz_c'] = img.affine.dot(np.hstack((np.array(img.shape[:3]) / 2.0, [1])))[:3]

    mapped_data = map_image(img, h1.get_affine(), h1.get_data_shape(), order=order)

    if imagetype == 'image':
        background = get_largest_cc(mapped_data == 0)
        if background[0,0,0]:
            mapped_data[background == 1] = -4.0
        else:
            mapped_data[background == 0] = -4.0

    if rescale:

        src_min, scale = getscale(img.get_data(), 0, 255)
        mapped_data = scalecrop(mapped_data, 0, 255, src_min, scale)

    new_data = np.float32(mapped_data)
    new_data[np.isnan(new_data)] = 0.0
    new_img = nib.MGHImage(new_data, h1.get_affine(), h1)

    new_img.set_data_dtype(np.float32)

    return new_img

def get_largest_cc(segmentation):
    from skimage.measure import label
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc

def is_conform(img, eps=1e-06):
    ishape = img.shape

    if len(ishape) > 3 and ishape[3] != 1:
        sys.exit('ERROR: Multiple input frames (' + format(img.shape[3]) + ') not supported!')

    if ishape[0] != 256 or ishape[1] != 256 or ishape[2] != 256:
        return False

    izoom = img.header.get_zooms()
    if izoom[0] != 1.0 or izoom[1] != 1.0 or izoom[2] != 1.0:
        return False

    iaffine = img.affine[0:3, 0:3] + [[1, 0, 0], [0, 0, -1], [0, 1, 0]]

    if np.max(np.abs(iaffine)) > 0.0 + eps:
        return False

    return True

def check_affine_in_nifti(img, logger=None):
    check = True
    message = ""

    if img.header['qform_code'] != 0 and np.max(np.abs(img.get_sform() - img.get_qform())) > 0.001:

        message = ' # qform and sform transform are not identical!'
        img.set_sform(img.get_qform())
        img.update_header()

    else:

        vox_size_head = img.header.get_zooms()
        aff = img.affine
        xsize = np.sqrt(aff[0][0] * aff[0][0] + aff[1][0] * aff[1][0] + aff[2][0] * aff[2][0])
        ysize = np.sqrt(aff[0][1] * aff[0][1] + aff[1][1] * aff[1][1] + aff[2][1] * aff[2][1])
        zsize = np.sqrt(aff[0][2] * aff[0][2] + aff[1][2] * aff[1][2] + aff[2][2] * aff[2][2])

        if (abs(xsize - vox_size_head[0]) > .001) or (abs(ysize - vox_size_head[1]) > .001) or (abs(zsize - vox_size_head[2]) > 0.001):
            message = "#############################################################\n" \
                      "ERROR: Invalid Nifti-header! Affine matrix is inconsistent with Voxel sizes. " \
                      "\nVoxel size (from header) vs. Voxel size in affine: " \
                      "({}, {}, {}), ({}, {}, {})\nInput Affine----------------\n{}\n" \
                      "#############################################################".format(vox_size_head[0],
                                                                                             vox_size_head[1],
                                                                                             vox_size_head[2],
                                                                                             xsize, ysize, zsize,
                                                                                             aff)
            check = False

    if logger is not None:
        logger.info(message)

    else:
        print(message)

    return check
