import argparse
import logging
from numpy import ndarray
from skimage import io, transform
import numpy as np
from scipy import optimize
from functools import partial
from typing import List
from joblib import Parallel, delayed
from pathlib import Path
import pickle


def parseargs():
    parser = argparse.ArgumentParser(
        description='Takes an input image and produces a tessellated version.'
    )
    parser.add_argument("-v", "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument("--input-file", help="Path to the image to process",
                        required=True)
    parser.add_argument("--output-dir", help="Where to store the output images",
                        required=True)
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    return args


def _find_closest_idx(h: int, w: int, centers: ndarray) -> int:
    x_y_array = np.ones((2, centers.shape[1]))
    x_y_array[0] *= h
    x_y_array[1] *= w
    tmp_ndarray = np.square(x_y_array - centers)
    square_sum = tmp_ndarray[0] + tmp_ndarray[1]
    return int(np.argmin(square_sum))


def calc_error(input_img: ndarray, output_img: ndarray) -> float:
    return np.absolute(input_img - output_img).sum() / np.ma.size(input_img)


def _get_centers_from_x_y_array(one_d_array: List) -> ndarray:
    to_reshape = np.asarray(one_d_array)
    if to_reshape.shape[0] % 2 != 0:
        raise RuntimeError('input must have an even number of rows')
    return np.reshape(to_reshape, (2, -1))


def make_tess_img(input_img: ndarray, one_d_array: ndarray) -> ndarray:
    one_d_array = one_d_array.copy()
    height = input_img.shape[0]
    width = input_img.shape[1]

    centers = _get_centers_from_x_y_array(one_d_array)
    num_points = centers.shape[1]
    centers[0] *= height
    centers[1] *= width

    closet_idx_mat = np.zeros((height, width), dtype=np.int32)
    center_colors = np.zeros((num_points, input_img.shape[2]))
    center_pixel_count = np.zeros(num_points)
    for h in range(height):
        for w in range(width):
            close_idx = _find_closest_idx(h, w, centers)
            # add the color to that center point
            center_colors[close_idx] += input_img[h, w]
            center_pixel_count[close_idx] += 1
            closet_idx_mat[h, w] = close_idx

    for center in range(num_points):
        center_colors[center] /= center_pixel_count[center]

    output_img = input_img.copy()
    for h in range(height):
        for w in range(width):
            output_img[h, w] = center_colors[closet_idx_mat[h, w]]
    return output_img


def get_error_from_one_d_array(input_img: ndarray, one_d_array: ndarray) -> float:
    output_img = make_tess_img(input_img.copy(), one_d_array.copy())
    error = calc_error(input_img.copy(), output_img.copy())
    return error


def update_1d_array(one_d_array: List, new_pt: List) -> ndarray:
    half_len = int(len(one_d_array)/2)
    in_array = one_d_array.copy()
    return np.append(np.append(in_array[:half_len], new_pt[0]),
                     np.append(in_array[half_len:], new_pt[1]))


def get_error_from_one_more_point(input_img: ndarray, one_d_array: List, new_pt: List) -> float:
    new_array = update_1d_array(one_d_array, new_pt)
    return get_error_from_one_d_array(input_img, new_array)


def make_x_y_list(num_pairs: int) -> List[ndarray]:
    new_pairs = []
    for _ in range(num_pairs):
        new_pairs.append(np.random.rand(2))
    return new_pairs


def main():
    args = parseargs()
    output_dir = Path.cwd() / args.output_dir
    input_img_full: ndarray = io.imread(args.input_file)
    input_img = transform.rescale(
        input_img_full, 0.2, multichannel=True, anti_aliasing=True)

    f_name = 'tmp_best.pickle'
    nun_cores = 10

    # num_points = 100
    # bounds = [(0, 1) for _ in range(2*num_points)]
    # # print(bounds)
    # cal = partial(get_error_from_one_d_array, input_img)
    # # print(cal(one_d_array))
    #
    # num_jobs = num_cores
    # all_results = Parallel(n_jobs=nun_cores)(delayed(
    #     optimize.dual_annealing)(cal, bounds, maxiter=1, seed=(1000+i*10)) for i in range(nun_jobs))
    #
    # # results = optimize.dual_annealing(cal, bounds, maxiter=1)
    # # print(results)
    # print([r.fun for r in all_results])
    # best_result = None
    # best_score = 100000
    # for result in all_results:
    #     if result.fun < best_score:
    #         best_result = result
    #         best_score = result.fun
    #
    # print(best_result)
    # with open(f_name, 'wb') as f:
    #     pickle.dump(best_result, f)

    # Start here.
    with open(f_name, 'rb') as f:
        best_result = pickle.load(f)
    output_img = make_tess_img(input_img_full, best_result.x)
    output_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    file_format = 'frame_{:04d}.png'
    io.imsave(output_dir / file_format.format(i), output_img)

    centers_so_far = best_result.x
    num_additional_frames = 1000
    # one_center_bound = [(0, 1) for _ in range(2)]
    num_new_pt_trials = 1000
    for f in range(1, num_additional_frames+1):
        print('starting frame {}'.format(f))
        x_y_list = make_x_y_list(num_new_pt_trials)
        one_point_calc = partial(get_error_from_one_more_point, input_img, centers_so_far)

        # all_results = Parallel(n_jobs=nun_cores)(delayed(
        #     optimize.dual_annealing)(
        #     one_point_calc, one_center_bound, maxiter=1, seed=(1000+i*10)) for i in range(1000))
        # best_result = None
        # best_score = 100000
        # for result in all_results:
        #     if result.fun < best_score:
        #         best_result = result
        #         best_score = result.fun
        # print(best_result)
        # centers_so_far = update_1d_array(centers_so_far, best_result.x)
        # output_img = make_tess_img(input_img_full, centers_so_far)
        # io.imsave(output_dir / 'frame_{}.jpg'.format(f), output_img)

        all_scores = Parallel(n_jobs=nun_cores)(delayed(
            one_point_calc)(x_y_list[i]) for i in range(num_new_pt_trials))

        best_result = None
        best_score = 100000
        for i in range(num_new_pt_trials):
            if all_scores[i] < best_score:
                best_result = x_y_list[i]
                best_score = all_scores[i]
        print(best_result, best_score)
        centers_so_far = update_1d_array(centers_so_far, list(best_result.tolist()))
        output_img = make_tess_img(input_img_full, centers_so_far)
        io.imsave(output_dir / file_format.format(f), output_img)


if __name__ == '__main__':
    main()
