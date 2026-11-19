"""2-D dead reckoning from IMU gyro + foot-contact kinematics, with map scan matching.

No global position, ground-truth velocity or MuJoCo object coordinates enter here.
This is an experimental proprioceptive laser-localization baseline, NOT complete
camera visual SLAM (which requires feature tracking, triangulation and loop closure).
"""
import math

import numpy as np

from .explorer import OccupancyMemory, SCAN_ANGLES


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotate(yaw, point):
    c,s=math.cos(yaw),math.sin(yaw)
    p=np.asarray(point,dtype=float)
    return np.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),axis=-1)


class PlanarEstimator:
    """Known spawn pose + imperfect IMU yaw and stance-foot odometry.

    Reference pose is an initial world convention, not continuous ground truth.
    Estimated position drifts with slip; scan matching is an optional correction.
    """
    def __init__(self, spawn=(0,0,0), matching=True):
        self.pose=np.asarray(spawn,dtype=float).copy()
        if self.pose.shape!=(3,) or not np.isfinite(self.pose).all():
            raise ValueError('Invalid initial pose')
        self.matching=bool(matching)
        self.prev_feet=None
        self.prev_contacts=None
        self.foot_updates=0
        self.fallback_updates=0
        self.scan_corrections=0
        self.last_match_residual=None

    def integrate(self, gyro_z, feet, contacts, dt, fallback_command=(0,0,0), fallback_speed=.13):
        if not (0<dt<=.1 and np.isfinite(gyro_z)):
            raise ValueError('Invalid IMU sample/time')
        f=np.asarray(feet,float);touch=np.asarray(contacts,bool)
        if f.shape!=(6,2) or touch.shape!=(6,) or not np.isfinite(f).all():
            raise ValueError('Invalid foot encoder/contact packet')
        prev_yaw=float(self.pose[2])
        new_yaw=wrap(prev_yaw+float(gyro_z)*dt)
        change=None
        if self.prev_feet is not None:
            stable=touch&self.prev_contacts
            if np.count_nonzero(stable)>=2:
                # Each stance foot is fixed in world frame (if no slip). Its
                # observed position in robot frame changes as the trunk moves.
                d=rotate(prev_yaw,self.prev_feet[stable])-rotate(new_yaw,f[stable])
                distances=np.linalg.norm(d,axis=1)
                valid=d[(distances<.025)]
                if len(valid)>=2:
                    change=np.median(valid,axis=0)
                    self.foot_updates+=1
        if change is None:
            cmd=np.asarray(fallback_command,dtype=float)
            if cmd.shape!=(3,) or not np.isfinite(cmd).all():
                raise ValueError('Invalid fallback command')
            # Gait speed is intentionally approximate, not MuJoCo qvel.
            change=rotate((prev_yaw+new_yaw)/2,cmd[:2])*fallback_speed*dt
            self.fallback_updates+=1
        self.pose[:2]+=np.clip(change,-.025,.025)
        self.pose[2]=new_yaw
        self.prev_feet=f.copy()
        self.prev_contacts=touch.copy()
        return self.pose.copy()

    def correct_with_map(self, memory, distances, hits, window_xy=.045,window_yaw=.035):
        """Local scan-to-occupancy alignment; does not use ground-truth pose.

        Require multiple previously mapped obstacles and current reflections.
        Conservative offsets avoid catastrophically distorting a sparse map.
        """
        if not self.matching:return self.pose.copy()
        if not isinstance(memory,OccupancyMemory):raise ValueError('Expected occupancy memory')
        d=np.asarray(distances,float);h=np.asarray(hits,bool)
        if d.shape!=SCAN_ANGLES.shape or h.shape!=d.shape:
            raise ValueError('Wrong scan dimensions')
        known=np.argwhere(memory.cells==memory.OCCUPIED)
        good=h&(d>.10)&(d<1.13)
        if len(known)<8 or np.count_nonzero(good)<3:
            return self.pose.copy()
        # The map is compact; subsampling avoids counting the same boundary
        # cell dozens of times. No scipy or simulator geometry required.
        known_world=np.stack([memory.center(tuple(cell)) for cell in known[::2]],axis=0)
        angle=SCAN_ANGLES[good]
        distance=d[good]
        prior=self.pose.copy()
        best=None;best_score=-np.inf
        for dx in (-window_xy,0,window_xy):
            for dy in (-window_xy,0,window_xy):
                for dyaw in (-window_yaw,0,window_yaw):
                    px=prior[0]+dx;py=prior[1]+dy;heading=prior[2]+dyaw
                    ends=np.stack([px+distance*np.cos(heading+angle),
                                   py+distance*np.sin(heading+angle)],axis=1)
                    nearest=np.min(np.sum((ends[:,None,:]-known_world[None,:,:])**2,axis=2),axis=1)
                    clipped=np.minimum(nearest,.13**2)
                    # Do not snap to existing data based on one ambiguous hit.
                    inliers=np.count_nonzero(nearest<.105**2)
                    score=-float(np.mean(clipped))-.001*(dx*dx+dy*dy)-.002*dyaw*dyaw
                    if inliers>=3 and score>best_score:
                        best=(dx,dy,dyaw,float(np.sqrt(np.mean(clipped))))
                        best_score=score
        if best is not None:
            dx,dy,dyaw,residual=best
            self.last_match_residual=residual
            # Only corrections with measured improvement over the no-shift
            # hypothesis, to prevent runaway self-reinforcing map alignment.
            heading=prior[2]
            zeros=np.stack([prior[0]+distance*np.cos(heading+angle),
                            prior[1]+distance*np.sin(heading+angle)],axis=1)
            base_nearest=np.min(np.sum((zeros[:,None,:]-known_world[None,:,:])**2,axis=2),axis=1)
            base_score=-float(np.mean(np.minimum(base_nearest,.13**2)))
            if best_score>base_score+.00012:
                self.pose+=np.array([dx,dy,dyaw])
                self.pose[2]=wrap(self.pose[2])
                self.scan_corrections+=1
        return self.pose.copy()
