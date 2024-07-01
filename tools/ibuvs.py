import numpy as np
from numpy import ndarray as NDA
from enum import Enum, auto
from typing import Union, Optional
# from utils.perception import CameraIntrinsic




def gaussian_kernel(e, bw):
    return np.exp(-0.5 * e**2 / bw**2)


def ibvs_mean(
    kp_cur: NDA, 
    Z_cur: Union[NDA, float], 
    kp_tar: NDA, 
    Z_tar: Union[NDA, float], 
    cam_intr, 
    mask: Optional[NDA] = None,
    algorithm="mean"
) -> NDA:
    """Image-based visual servo controller.

    Arguments:
    - kp_cur: (N, 2), 2 represents (x, y), feature points in pixel coordinate
    - Z_cur: (N,), depth of feature points in current camera frame
    - kp_tar: (N, 2), feature points in pixel coordinate
    - Z_tar: (N,), depth of feature points in target camera frame
    - cam_intr: camera intrinsic
    - mask: valid observation mask of kp_cur
    - algorithm: str, can be: mean, cur, tar

    Returns:
    - vel: (6,), [tx, ty, tz, wx, wy, wz], camera velocity in current camera frame
    - L_mean: (N*2, 6), jacobian, d(error)/dt = - L_mean @ vel
    """

    assert kp_cur.shape == kp_tar.shape, "number of feature points not match"
    num_fp = kp_cur.shape[0]

    fx, fy = cam_intr[0,0], cam_intr[1,1]
    cx, cy = cam_intr[0,2], cam_intr[1,2]

    u_cur = kp_cur[:, 0]
    v_cur = kp_cur[:, 1]
    # build interaction matrix at current camera frame
    L_cur = np.zeros((num_fp * 2, 6))
    L_cur[0::2, 0] = -fx / Z_cur
    L_cur[0::2, 2] = (u_cur - cx) / Z_cur
    L_cur[0::2, 3] = (u_cur - cx) * (v_cur - cy) / fy
    L_cur[0::2, 4] = -(fx + (u_cur - cx)**2 / fx)
    L_cur[0::2, 5] = fx / fy * (v_cur - cy)
    L_cur[1::2, 1] = -fy / Z_cur
    L_cur[1::2, 2] = (v_cur - cy) / Z_cur
    L_cur[1::2, 3] = fy + (v_cur - cy)**2 / fy
    L_cur[1::2, 4] = -(u_cur - cx) * (v_cur - cy) / fx
    L_cur[1::2, 5] = -fy / fx * (u_cur - cx)

    u_tar = kp_tar[:, 0]
    v_tar = kp_tar[:, 1]
    # build interaction matrix at target camera frame
    L_tar = np.zeros((num_fp * 2, 6))
    L_tar[0::2, 0] = -fx / Z_tar
    L_tar[0::2, 2] = (u_tar - cx) / Z_tar
    L_tar[0::2, 3] = (u_tar - cx) * (v_tar - cy) / fy
    L_tar[0::2, 4] = -(fx + (u_tar - cx)**2 / fx)
    L_tar[0::2, 5] = fx / fy * (v_tar - cy)
    L_tar[1::2, 1] = -fy / Z_tar
    L_tar[1::2, 2] = (v_tar - cy) / Z_tar
    L_tar[1::2, 3] = fy + (v_tar - cy)**2 / fy
    L_tar[1::2, 4] = -(u_tar - cx) * (v_tar - cy) / fx
    L_tar[1::2, 5] = -fy / fx * (u_tar - cx)

    assert algorithm in ["cur", "tar", "mean"]
    if algorithm == "mean":
        L_mean = (L_cur + L_tar) / 2.0
        # L_mean = L_cur * 0.25 + L_tar * 0.75
    elif algorithm == "cur":
        L_mean = L_cur
    elif algorithm == "tar":
        L_mean = L_tar

    mask_rep = None if mask is None else np.repeat(mask, 2)
    if mask_rep is not None:
        L_mean[~mask_rep] = L_tar[~mask_rep]

    error = np.zeros(num_fp * 2)
    error[0::2] = u_tar - u_cur
    error[1::2] = v_tar - v_cur
    if mask_rep is not None:
        error[~mask_rep] = 0

    invL = np.linalg.pinv(L_mean)
    vel = np.dot(invL, error)
    return vel, L_mean


def vec2skew(v: NDA):
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])


def calc_Jcr(tcT: NDA):
    """
    Arguments:
    - tcT: (4, 4), camera pose in tcp coordinate

    Returns:
    - Jcr: (6, 6), v_c = J_cr v_r
    """

    # c: camera coordinate; t: tcp coordinate
    # \dot{e} = J_{ec} v_c
    # v_c = J_{cr} v_r

    # ^{i+1}v_{i+1} = ^{i+1}_{i}R @ (^{i}v_{i} + ^{i}w_{i} \times ^{i}P_{i+1})
    # ^{i+1}w_{i+1} = ^{i+1}_{i}R @ ^{i}w_{i}
    # ------------------------------------------------------------------------
    # ^{c}v_{c} = ^{c}_{t}R @ (^{t}v_{t} + ^{t}w_{t} \times ^{t}P_{c})
    # ^{c}w_{c} = ^{c}_{t}R @ ^{t}w_{t}
    # 
    # => ^{c}v_{c} = ^{c}_{t}R @ ^{t}v_{t} - ^{c}_{t}R @ skew(^{t}P_{c}) @ ^{t}w_{t}
    # => [[^{c}v_{c}]  = [[ ^{c}_{t}R, - ^{c}_{t}R @ skew(^{t}P_{c}) ]  @ [[^{t}v_{t}]
    #     [^{c}w_{c}]]    [         0,                     ^{c}_{t}R ]]    [^{t}w_{t}]]
    ctR = np.linalg.inv(tcT[:3, :3])
    tPc_skew = vec2skew(tcT[:3, 3])
    J_cr = np.kron(np.eye(2), ctR)
    J_cr[:3, 3:] = -ctR @ tPc_skew
    return J_cr


class IBUVS(object):
    def __init__(self, cam_intr, tcT: NDA, method):
        """
        - cam_intr: camera intrinsic, can be estimated
        - tcT: shape=(4, 4), camera pose in tcp frame, can be estimated
        - method: can be ANALYTICAL or KF
        """
        self.cam_intr = cam_intr
        self.tcT = tcT
        self.method = method
        self.J_cr = calc_Jcr(tcT)
    
    def set_desired(self, desired_kp: NDA, desired_Z: Union[NDA, float]):
        self.desired_kp = desired_kp
        self.desired_Z = desired_Z
        self.num_kp = len(desired_kp)

        N, M = self.num_kp*2, 6

        self.X = np.zeros(N*M)
        self.Z = np.zeros(N)
        self.H = np.zeros((N, N*M))
        self.P = np.eye(N*M) * 1  # state covariance
        self.K = np.zeros((N*M, N))
        # self.Q = np.eye(N*M) * 1e-3  # motion noise
        self.Q = np.eye(N*M) * 7**2  # motion noise, 28 for 0.25cur + 0.75tar, 8 for half-half
        self.R = np.eye(N) * 0.5**2  # measure noise

        self.accum_time = 0
        self.first_run = True

    def _init_jac(self, initial_kp: NDA, mask: Optional[NDA] = None) -> NDA:
        """
        Arguments:
        - initial_kp: (M, 2), float, aligned with desired_kp, 
            kp on pixel coordinate, 0~W or H
        - initial_btT: (4, 4), tcp pose in base coordinate
        - mask: valid mask

        Returns:
        - J_er: (M*2, 6), d(error)/dt = J_er * v_r
        """
        _, J_ec = ibvs_mean(
            initial_kp, self.desired_Z, 
            self.desired_kp, self.desired_Z, 
            self.cam_intr, mask,
            algorithm="mean"
        )
        J_er = J_ec @ self.J_cr
        return J_er

    def estimate(self, current_kp: NDA, prev_actual_vt, dt, mask: NDA = None):
        """
        Arguments:
        - current_kp: (M, 2), float, points on pixel coordinate, 0~W or H
        - prev_actual_vt: (6,), [vx, vy, vz, wx, wy, wz], previous tcp velocity in tcp coordinate
            for actual excution (because the raw prediction can be modified)
        - dt: float, how long to conduct the `prev_vt`
        - mask: (M,), bool

        Returns:
        - vt: (6,), [vx, vy, vz, wx, wy, wz], tcp velocity in tcp coordinate
        """
        if self.method == "analytical":
            vel_cam, J_ec = ibvs_mean(
                current_kp, self.desired_Z, 
                self.desired_kp, self.desired_Z, 
                self.cam_intr, mask, "mean")
            # vel_cam = J_cr @ vel_tcp => vel_tcp = J_cr ^{-1} @ vel_cam
            vel_tcp = np.linalg.inv(self.J_cr) @ vel_cam
            return vel_tcp


        N, M = self.num_kp*2, 6

        if mask is None:
            # assume all valid
            mask = np.ones(self.num_kp, dtype=bool)
        mask_rep = np.repeat(mask, 2)

        if self.first_run:
            self.first_run = False
            J_er = self._init_jac(current_kp, mask)
            self.X[:] = J_er.ravel()  # init state
            self.previous_kp = current_kp.copy()
            self.previous_mask_rep = mask_rep.copy()

        ### KF based method ###
        # prediction
        # J_er = self._init_jac(current_kp, mask)
        # self.X = self.X  # no motion
        # self.X = self.X + 0.1 * (J_er.ravel() - self.X)

        _, J_ec = ibvs_mean(
            current_kp, self.desired_Z,
            self.desired_kp, self.desired_Z,
            self.cam_intr, mask,
            algorithm="mean"
        )
        J_er = J_ec @ self.J_cr
        self.X = (self.X + J_er.ravel()) / 2.0
        # self.X = self.X  # no motion
        self.P = self.P + self.Q

        # measurement
        self.Z[:] = current_kp.ravel() - self.previous_kp.ravel()
        self.Z[~(mask_rep & self.previous_mask_rep)] = 0
        obs_mat = np.diag(mask_rep & self.previous_mask_rep).astype(float)
        self.H = np.kron(obs_mat, prev_actual_vt * dt)

        # update
        self.K = self.P @ self.H.T @ np.linalg.pinv(self.H @ self.P @ self.H.T + self.R)
        self.X = self.X + self.K @ (self.Z - self.H @ self.X)
        self.P = (np.eye(N*M) - self.K @ self.H) @ self.P

        J_er = self.X.reshape((N, M))
        error = current_kp.ravel() - self.desired_kp.ravel()
        error[~mask_rep] = 0.0
        vt = -np.linalg.pinv(J_er) @ error

        self.previous_kp = current_kp.copy()
        self.previous_mask_rep = mask_rep.copy()

        return vt

