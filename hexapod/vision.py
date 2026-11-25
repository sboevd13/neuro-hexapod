"""Monocular perception for laboratory courses. Training and runtime use identical optics."""
from pathlib import Path
import numpy as np
from .perception import Pinhole, level_rotation
from .ports import Scan

RGB_WIDTH,RGB_HEIGHT=80,48
MODEL=Path(__file__).resolve().parent/'weights'/'camera_rgb.npz'
CLASSES=('clear','roof','corridor','terrain','barrier')

def features(image,attitude):
    image=np.asarray(image)
    if image.shape!=(RGB_HEIGHT,RGB_WIDTH,3) or image.dtype!=np.uint8:
        raise ValueError('Expected calibrated 80 x 48 uint8 RGB frame')
    tiles=image.reshape(12,4,20,4,3).mean(axis=(1,3))/255.
    luma=tiles.mean(axis=2)
    chroma=(tiles[:,:,0]-tiles[:,:,2])/(.18+luma)
    return np.r_[luma.ravel(),chroma.ravel(),np.array(attitude[:2])*4].astype(np.float32)

class CorridorEdges:
    """Locate the two floor-to-wall boundaries along calibrated ground-plane rays."""

    def __init__(self):
        camera = Pinhole(RGB_WIDTH, RGB_HEIGHT, fovy=85.)
        self.rays = camera.rays.reshape(RGB_HEIGHT, RGB_WIDTH, 3)

    def left_fraction(self, image, attitude):
        """Fraction of the predicted opening on the left, independent of camera height."""
        rays = self.rays @ level_rotation(*attitude[:2]).T
        center = RGB_WIDTH // 2
        rgb = image.astype(np.float32)
        intensity = rgb.sum(axis=2)
        chroma = rgb / np.maximum(intensity[..., None], 1.)
        floor = np.median(chroma[:, center-3:center+4], axis=1)
        different = (np.linalg.norm(chroma-floor[:, None], axis=2) > .10) & (intensity > 35)
        # A run of three pixels rejects checkerboard seams and sensor noise.
        solid = different[:, :-2] & different[:, 1:-1] & different[:, 2:]
        fractions = []
        for v in range(RGB_HEIGHT // 2, RGB_HEIGHT-1, 2):
            slope = -rays[v, center, 2] / rays[v, center, 0]
            if not .15 < slope < .75:
                continue
            left_pixels = np.flatnonzero(solid[v, :center-4])
            right_pixels = np.flatnonzero(solid[v, center+3:])
            if not left_pixels.size or not right_pixels.size:
                continue
            left_u = left_pixels[-1] + 2
            right_u = right_pixels[0] + center + 3
            left_ray, right_ray = rays[v, [left_u, right_u]]
            if min(-left_ray[2], -right_ray[2]) <= .05:
                continue
            # Intersect each ray with a plane one camera-height below the lens.
            left = left_ray[1] / -left_ray[2]
            right = -right_ray[1] / -right_ray[2]
            width = left + right
            if min(left, right) > .45 and 2.25 < width < 5.55:
                fractions.append(left/width)
        if len(fractions) < 2:
            return None
        fractions = np.array(fractions)
        fraction = float(np.median(fractions))
        if np.median(abs(fractions-fraction)) > .055:
            return None
        return float(np.clip(fraction, .30, .70))


class MonoPerception:
    def __init__(self,path=MODEL):
        self.corridor_edges = CorridorEdges()
        with np.load(path,allow_pickle=False) as data:
            self.layers=[(data[f'w{i}'],data[f'b{i}']) for i in range(3)]
            self.mean,self.scale=data['mean'],data['scale']

    def read(self,image,attitude,time):
        if image is None or np.ptp(image)<5:
            return Scan(time,valid=False)
        x=(features(image,attitude)-self.mean)/self.scale
        for w,b in self.layers[:-1]:
            x=np.maximum(0.,x@w+b)
        y=x@self.layers[-1][0]+self.layers[-1][1]
        k=int(np.argmax(y[:5]))
        y[5:]/=max(.30,float(y[k]))
        if k==1:
            return Scan(time,roof_m=float(np.clip(y[5]*.25,.08,.22)))
        if k==2:
            left,right=np.clip(y[6:8]*.35,.08,.34)
            fraction = self.corridor_edges.left_fraction(image, attitude)
            if fraction is not None:
                width = left+right
                left, right = width*fraction, width*(1-fraction)
            return Scan(time,left_m=float(left),right_m=float(right))
        if k==3:
            return Scan(time,ground_m=tuple(np.clip(y[8:11]*.06,0.,.055).tolist()))
        if k==4:
            return Scan(time,wall_m=float(np.clip(y[11]*.5,.02,.39)))
        return Scan(time)

def label(scan):
    y=np.zeros(12,dtype=np.float32)
    if ((scan.wall_m is not None and scan.wall_m<.41) or (scan.roof_m is not None and scan.roof_m<.132)
            or (scan.left_m and scan.right_m and scan.left_m+scan.right_m<.387)):
        k=4
        y[11]=(.25 if scan.wall_m is None else scan.wall_m)/.5
    elif scan.roof_m is not None and scan.roof_m<.195:
        k=1
        y[5]=scan.roof_m/.25
    elif scan.left_m and scan.right_m and scan.left_m+scan.right_m<.50:
        k=2
        y[6:8]=np.array([scan.left_m,scan.right_m])/.35
    elif max(scan.ground_m)>.009:
        k=3
        y[8:11]=np.array(scan.ground_m)/.06
    else:
        k=0
    y[k]=1.
    return y
