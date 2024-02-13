def thresholding(lbl, thresh=0.5):
    for i in range(0, lbl.shape[0]):
        for j in range(0, lbl.shape[1]):
            for k in range(0, lbl.shape[2]):
                if lbl[i][j][k] > thresh:
                    lbl[i][j][k] = 1
                else:
                    lbl[i][j][k] = 0
    return lbl
