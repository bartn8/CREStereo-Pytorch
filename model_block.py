import importlib
cre_stereo = importlib.import_module("thirdparty.CREStereo-Pytorch.nets.crestereo")
#cre_stereo.CREStereo

import torch
import torch.nn.functional as F

import numpy as np
import cv2

class CREStereoBLock:
    def __init__(self, inference_size = [1024,1536], iters=20,
     max_disparity=256, mixed_precision=False, device = "cpu", verbose=False): 
        
        self.logName = "CREStereo Pytorch Block"
        self.verbose = verbose
        
        self.inference_size = inference_size
        self.n_iter = iters
        self.max_disp = max_disparity # Useless parameter (not used in net)
        self.mixed_precision = mixed_precision
        self.device = device
        
        self.disposed = False

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        self.log(f"Building Model...")

        self.model = cre_stereo.CREStereo(max_disp=self.max_disp, mixed_precision=self.mixed_precision, test_mode=True)

    def load(self, model_path):
        self.log("Loading frozen model")
        self.log(f"Model checkpoint path: {model_path}")

        pretrained_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(pretrained_dict)

        self.model.to(self.device)
        self.model.eval()


    def dispose(self):
        if not self.disposed:
            del self.model
            self.disposed = True

    def _conv_image(self,img):
        
        if self.inference_size is None:
            h,w = img.shape[:2]
            pad_ht = (((h // 32) + 1) * 32 - h) % 32
            pad_wd = (((w // 32) + 1) * 32 - w) % 32
            _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
            img = cv2.copyMakeBorder(img, _pad[2], _pad[3], _pad[0], _pad[1], cv2.BORDER_REPLICATE)
        else:
            eval_w, eval_h = self.inference_size[:2]
            img = cv2.resize(img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
            _pad = None

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # else:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.transpose(2,0,1)
        img = np.ascontiguousarray(img[None, :,:,:])
        img = torch.tensor(img.astype("float32")).to(self.device)

        img_dw2 = F.interpolate(
                img,
                size=(img.shape[2] // 2, img.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
            )

        return img, img_dw2, _pad

    
    def test(self, left_vpp, right_vpp):
        #Input conversion
        in_h, in_w = left_vpp.shape[:2]
        
        

        left_vpp, left_vpp_dw2, _pad = self._conv_image(left_vpp)
        right_vpp, right_vpp_dw2,_ = self._conv_image(right_vpp)

        with torch.no_grad():
            pred_flow_dw2 = self.model(left_vpp_dw2, right_vpp_dw2, iters=self.n_iter, flow_init=None)
            pred_flow = self.model(left_vpp, right_vpp, iters=self.n_iter, flow_init=pred_flow_dw2)
        
        pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

        if self.inference_size is None:
            ht, wd = pred_disp.shape[-2:]
            c = [_pad[2], ht-_pad[3], _pad[0], wd-_pad[1]]
            disp = pred_disp[..., c[0]:c[1], c[2]:c[3]]
        else:
            eval_w = self.inference_size[0]
            t = float(in_w) / float(eval_w)
            disp = cv2.resize(pred_disp, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

        return disp
            