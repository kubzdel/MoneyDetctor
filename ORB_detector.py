from skimage.feature import ORB, match_descriptors

def exec(img1, img2, name):
    det_extr1 = ORB(n_keypoints = 500)
    det_extr2 = ORB(n_keypoints = 500)
    det_extr1.detect_and_extract(img1)
    det_extr2.detect_and_extract(img2)
    matches = match_descriptors(det_extr1.descriptors, det_extr2.descriptors)
    print(name, "matching points " ,len(matches))