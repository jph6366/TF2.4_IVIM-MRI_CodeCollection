import numpy as np
import nibabel
from wrappers.OsipiBase import OsipiBase
from standardized.IAR_LU_biexp import IAR_LU_biexp


def preprocess(image, bvals, voxel=None, direction=None):
    
    if voxel is not None and image.ndim == 4:
        x, y, z = voxel
        signal = np.squeeze(image[x, y, z, :])
    else:
        # Assume image is already a 1D signal
        signal = np.squeeze(image)


    # normalize data
    selsb = np.array(bvals) == 0
    if np.any(selsb):
        S0 = np.nanmean(signal[selsb])
        if S0 > 0:
            signal = signal / S0
        else:
            signal = np.ones_like(signal)  # Fallback to avoid division by zero
    else:
        signal = np.insert(signal, 0, 1.0)  # Assume normalized signal starts with 1


    if direction is not None and image.ndim == 4:
        # Adjust for direction (assuming b-values repeat per direction)
        bvals_out = np.unique(bvals)[::6]
        bvals_out = np.insert(bvals_out, 0, 0)
        signal = signal[direction::6]
        signal = np.insert(signal, 0, 1.0)
    else:
        bvals_out = bvals

    return signal, bvals_out

def ivim_pipeline(bvec_path=None, bval_path=None, data_path=None, 
                  data_dict=None, voxel=None, direction=None, algorithm=None):
    algorithm = algorithm or IAR_LU_biexp()

    # Load data from files if provided
    if data_path and bval_path:
        bvals = np.genfromtxt(bval_path)
        data = nibabel.load(data_path)
        image = data.get_fdata()
    elif data_dict:
        # Use test data dictionary
        image = data_dict.get('data')
        bvals = data_dict.get('bvals', [])
    else:
        raise ValueError("Either data_path and bval_path or data_dict must be provided.")

    signal, bvals = preprocess(image, bvals, voxel=voxel, direction=direction)
    fit = OsipiBase(algorithm=algorithm)
    fit_result = fit.osipi_fit(signal, bvals)

    def to_list_if_needed(value):
        return value.tolist() if isinstance(value, np.ndarray) else value
    test_result = {
        "algorithm": str(algorithm),
        "f_fit": to_list_if_needed(fit_result['f']),
        "Dp_fit": to_list_if_needed(fit_result['Dp']),
        "D_fit": to_list_if_needed(fit_result['D']),
    }


    return test_result