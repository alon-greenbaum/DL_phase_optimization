import skimage.io
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import pylab as pl
import sys
#%matplotlib inline
def FWHM(path):
    #file_path = './results/model_fig2/gaussianprofile/'
    x = range(0,201)
    FWHM = []
    ans = []
    for index in range(len(x)):
        img = skimage.io.imread(path+str(x[index]) + '.tiff')
        
        intensity_values = sum(img.astype(np.uint32))
        #print(intensity_values)
        #print(x[index])
        ans.append(intensity_values)
        
    return ans

def cal_beamwidth(beam_inten):
    beam_width = []
    x  = range(41)
    for i in range(len(beam_inten)):
        inten = beam_inten[i]
        spline = UnivariateSpline(x, inten-np.max(inten)/2, s=0)
        r1, r2 = spline.roots()
        beam_width.append(r2-r1)
    return beam_width


if __name__ == '__main__':
    
    ##crop same size image
    img = skimage.io.imread('./paperfigure_2_22_23/fig2/experiment_3/gau_new_f15/515.008.tiff')
    x = 1681
    y = 1093
    d = 60
    img_small = img[x - d:x + d + 1,y - d:y + d + 1]
    skimage.io.imsave('./paperfigure_2_22_23/fig2/experiment_3/gau_test.tiff',img_small)
    sys.exit()
    
    # img = skimage.io.imread('./results/model_fig5/gau_-100um.tiff')
    # inten = sum(img) - min(sum(img)) 
    # x = range(61)
    # spline = UnivariateSpline(x, inten-np.max(inten)/2, s=0)
    # r = spline.roots()
    # print(r[-1] - r[0])
    #sys.exit()
    
    ##
    
    
    #gaussian = FWHM('./gaussian_folder/')
    gaussian = FWHM('./3D_folder/gau/')
    optimized = FWHM('./3D_folder/opt/')
    bessel = FWHM('./3D_folder/bessel/')
    gaussian_width = cal_beamwidth(gaussian)
    optimized_width = cal_beamwidth(optimized)
    bessel_width = cal_beamwidth(bessel)
    x_dist = range(-100,101)
    x = range(-20,21)
    
    plt.plot(x,gaussian[0]/max(gaussian[0]),'r',x,bessel[0]/max(bessel[0]),'b',x,optimized[0]/max(optimized[0]))
    plt.legend(['Gaussian light-sheet', 'bessel light-sheet', 'Butterfly light-sheet' ])
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    
    plt.rc('font',family = 'Times New Roman')
    plt.figure(figsize=(13, 6))
    plt.plot(x_dist,gaussian_width,'r',x_dist,optimized_width,'g')
    plt.legend(['Gaussian light-sheet','Optimized light-sheet'],fontsize = 30)
    #plt.title('Beam width comparsion',fontsize = 30)
    #plt.xlabel('Propagation distance')
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    plt.savefig('FWHM.png', bbox_inches='tight')
    #sys.exit()
    # # reference https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html
    ## https://itecnote.com/tecnote/python-finding-the-full-width-half-maximum-of-a-peak/
    
    
    x = range(-100,101)
    gau_inten = []
    for i in range(0,201):
        gau_inten.append(gaussian[i][20])
    opt_inten = []
    for i in range(0,201):
        opt_inten.append(optimized[i][20])
    plt.figure(figsize=(13, 6))
    plt.plot(x,gau_inten,'r',x,opt_inten,'g')
    plt.legend(['Gaussian light-sheet','Optimized light-sheet'],fontsize = 30,loc="best")
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    #plt.title('Beam intensity ',fontsize = 30)
    plt.savefig('intensity.png', bbox_inches='tight')
    
    x  = range(41)
    # for gaussian beam
    gau = gaussian[200]
    spline_gau = UnivariateSpline(x, gau-np.max(gau)/2, s=0)
    g1, g2 = spline_gau.roots()
    print('gaussian width: ', g2 - g1)
    # for optimized beam
    opt = optimized[200]
    spline_opt = UnivariateSpline(x, opt-np.max(opt)/2, s=0)
    r1, r2 = spline_opt.roots()
    print('opt width: ', r2 - r1)
    
    fig = pl.plot(x,gau,'g',linewidth=3)
    pl.axvspan(g1, g2, facecolor='g', alpha=0.3)
    pl.plot(x,opt,'r',linewidth=3)
    pl.axvspan(r1, r2, facecolor='r', alpha=0.3)
    pl.axis('on')
    plt.gca().axes.get_yaxis().set_visible(False)
    #plt.gca().axes.get_xaxis().set_visible(True)
    #plt.gca().axes.get_xaxis()
    plt.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    #pl.show()
    pl.savefig('abcd1.png',bbox_inches='tight', pad_inches=0)
    
    ## experiment calculate width
    x  = range(121)
    gau_img = skimage.io.imread('./paperfigure_2_22_23/fig2/experiment_3/gau_600um.tiff').astype(np.int32)
    opt_img = skimage.io.imread('./paperfigure_2_22_23/fig2/experiment_3/opt_600um.tiff').astype(np.int32)
    gau_int = sum(gau_img)/121
    opt_int = sum(opt_img)/121
    spline_gau = UnivariateSpline(x, gau_int-np.max(gau_int)*0.5, s=0)
    g1, g2 = spline_gau.roots()
    print('gaussian width: ', (g2 - g1)*0.48)

    spline_opt = UnivariateSpline(x, opt_int-np.max(opt_int)*0.5, s=0)
    r1, r2 = spline_opt.roots()
    print('opt width: ', (r2 - r1)*0.48)
    
    pl.plot(x,gau_int,'r',linewidth=3)
    pl.axvspan(g1, g2, facecolor='r', alpha=0.3)
    pl.plot(x,opt_int,'g',linewidth=3)
    pl.axvspan(r1, r2, facecolor='g', alpha=0.3)
    pl.axis('off')
    #pl.show()
    pl.savefig('abcd1.png',bbox_inches='tight', pad_inches=0)
    
    
    
    # binarize image
    img = skimage.io.imread('network_pred_output.tiff')
    img[img>0] = 1
    img[img<=0] = 0
    skimage.io.imsave('network_pred_threshold.tiff', img)
    
    
    
    
    x  = range(41)
    gau = gaussian[200]
    opt = optimized[200]
    pl.plot(x,gau,'r',linewidth=3)
    pl.plot(x,opt,'g',linewidth=3)
    pl.axis('off')
    pl.savefig('abcd1.png',bbox_inches='tight', pad_inches=0)
    
    
    
    
    
    

