
def calc_wetday_frac(da, thresh=0):
    p = (da > thresh)
    wdf = p.mean(dim='time')

    return wdf.where(wdf > 0)
