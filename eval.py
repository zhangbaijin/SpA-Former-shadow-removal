import numpy as np
#from skimage.measure import compare_ssim as SSIM
from skimage.metrics import structural_similarity as SSIM

from torch.autograd import Variable

from utils import save_image

from sklearn import metrics
import math
#from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage import color
import cv2



def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    mae = 0


    
    for i, batch in enumerate(test_data_loader):
        x, t = Variable(batch[0]), Variable(batch[1])
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        att, out = gen(x)
       
        if epoch % config.snapshot_interval == 0:
            h = 1
            w = 3
            c = 3
            p = config.width
            q = config.height

            allim = np.zeros((h, w, c, p, q))
            x_ = x.cpu().numpy()[0]
            t_ = t.cpu().numpy()[0]
            out_ = out.cpu().numpy()[0]
            in_rgb = x_[:3]
            t_rgb = t_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = t_rgb * 255
            
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*p, w*q, c))

            #save_image(config.out_dir, allim, i, epoch)
    

        mse = criterionMSE(out, t)
    
        psnr = 10 * np.log10(1 / mse.item())
       

        img1 = np.tensordot(out.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        img2 = np.tensordot(t.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        
        
        '''
        out = out.cpu().numpy()
        t = t.cpu().numpy()
        out_lab = cv2.cvtColor(out,cv2.COLOR_RGB2LAB)
        t_lab = cv2.cvtColor(t,cv2.COLOR_RGB2LAB)
 
        
        mae = np.mean(abs(out_lab - t_lab))
        '''
    
        ssim = SSIM(img1, img2)
        avg_mae += mae
        avg_psnr += psnr
        avg_ssim += ssim


        
    avg_mae = avg_mae / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    

    print("===> Avg. MAE: {:.4f}".format(avg_mae))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))


    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['mae'] = avg_mae
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim


    
    
    

    return log_test
