import math
from device import np, Array
from typing import Tuple, Union
from parser import tag


@tag()
class Transform:
    def __init__(
        self,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        scale: Tuple[float, float, float] | float
    ):
        self.position = position
        self.rotation = rotation
        self.scale = scale


identity_transform = Transform((0., 0., 0.), (0., 0., 0.), (1.))


def normalize(x: Array) -> Array:
    """
    L2归一化
    Args:
        x(Array): [3, N]
    """
    return x / np.sqrt((x ** 2).sum(axis=0, keepdims=True))


def normalize_homogeneous(x: Array) -> Array:
    """
    齐次坐标规范化，对齐次坐标的每个分量除以其最后一个分量
    Args:
        x(Array): [4, N]
    """
    return x / x[3, :]


def expand1(x: Array) -> Array:
    """
    将非齐次坐标转换为齐次坐标，最后一个分量补1
    Args:
        x(Array): [3, N]
    Returns:
        x(Array): [4, N]
    """
    return np.concat([x, np.ones((1, x.shape[-1]))], axis=0)


def expand0(x: Array) -> Array:
    """
    将非齐次坐标转换为齐次坐标，最后一个分量补0
    Args:
        x(Array): [3, N]
    Returns:
        x(Array): [4, N]
    """
    return np.concat([x, np.zeros((1, x.shape[-1]))], axis=0)


def translation(t: Union[Tuple[float, float, float], Array]) -> Tuple[Array, Array]:
    """
    Args:
        t(Tuple[float, float, float]): 新坐标系的原点在源坐标系下的坐标
    Returns:
        matrix(Array): 源坐标系到新坐标系的变换矩阵
        inv(Array): 新坐标系到源坐标系的变换矩阵
    """
    # 注意：t 可能是 cupy 数组（如 view_transform 传入的摄像机位置），
    # 不能把 cupy 标量放进 Python 列表再构造数组，用切片赋值保持计算在 GPU 上
    t_ = np.array(t, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, 3] = -t_
    inv = np.eye(4)
    inv[:3, 3] = t_
    return matrix, inv


def rotation(euler: Tuple[float, float, float]) -> Tuple[Array, Array]:
    """
    Args:
        euler(Tuple[float, float, float]): 新坐标系相对于源坐标系的欧拉角**（角度制）**\n
        每个分量是朝着旋转轴正方向看去，绕旋转轴逆时针旋转的度数（按照教程的约定）\n
        似乎与常见的约定不符，后续仍需确认
    Returns:
        matrix(Array): 源坐标系到新坐标系的变换矩阵
        inv(Array): 新坐标系到源坐标系的变换矩阵
    """
    cx = math.cos(euler[0] * np.pi / 180)
    sx = math.sin(euler[0] * np.pi / 180)
    cy = math.cos(euler[1] * np.pi / 180)
    sy = math.sin(euler[1] * np.pi / 180)
    cz = math.cos(euler[2] * np.pi / 180)
    sz = math.sin(euler[2] * np.pi / 180)
    # rx 是源坐标系中，朝着x轴正方向看去，点绕x轴逆时针旋转theta的旋转矩阵
    rx = np.array([
        [1., 0., 0., 0.],
        [0., cx, -sx, 0.],
        [0., sx, cx, 0.],
        [0., 0., 0., 1.]
    ])
    ry = np.array([
        [cy, 0., sy, 0.],
        [0., 1., 0., 0.],
        [-sy, 0., cy, 0.],
        [0., 0., 0., 1.]
    ])
    rz = np.array([
        [cz, -sz, 0., 0.],
        [sz, cz, 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    inv = rz @ ry @ rx
    return inv.T, inv


def scale(s: Union[Tuple[float, float, float], float]) -> Tuple[Array, Array]:
    """
    Args:
        s(Union[Tuple[float, float, float], float]): 新坐标系的基与源坐标系的基的模长之比\n
        记新坐标系的基为(x',y',z')，源坐标系的基为(x,y,z)，则s=(x'/x, y'/y, z'/z)
    Returns:
        matrix(Array): 源坐标系到新坐标系的变换矩阵
        inv(Array): 新坐标系到源坐标系的变换矩阵
    """
    if isinstance(s, float) or isinstance(s, int):
        s = (s, s, s)
    matrix = np.array([
        [1 / s[0], 0., 0., 0.],
        [0., 1 / s[1], 0., 0.],
        [0., 0., 1 / s[2], 0.],
        [0., 0., 0., 1.]
    ])
    inv = np.array([
        [s[0], 0., 0., 0.],
        [0., s[1], 0., 0.],
        [0., 0., s[2], 0.],
        [0., 0., 0., 1.]
    ])
    return matrix, inv


def world_transform(
    position: Tuple[float, float, float],
    euler: Tuple[float, float, float],
    s: Union[Tuple[float, float, float], float]
) -> Tuple[Array, Array]:
    """
    世界变换矩阵及其逆矩阵
    Args:
        position(Tuple[float, float, float]): 物体坐标系的原点在世界坐标系下的坐标
        euler(Tuple[float, float, float]): 物体坐标系相对于世界坐标系的欧拉角**（角度制）**
        s(Union[Tuple[float, float, float], float]): 物体自身的缩放
    Returns:
        matrix(Array): 物体坐标系到世界坐标系的变换矩阵
        inv(Array): 世界坐标系到物体坐标系的变换矩阵
    """
    T, T_inv = translation(position)
    R, R_inv = rotation(euler)
    S, S_inv = scale(s)
    # 理解：变换前的物体放在源坐标系中，坐标是(x,y,z)，依次经缩放、旋转、平移后得到变换后的物体，坐标是(x',y',z')
    # 即(x',y',z')^T = T_o @ R_o @ S_o @ (x,y,z)^T，其中T_o,R_o,S_o分别是点的平移、旋转、缩放矩阵
    # (x,y,z)同时也是变换后物体在物体坐标系中的坐标，(x',y',z')是物体变换后在世界坐标系中的坐标
    # 物体坐标系到世界坐标系的坐标变换矩阵是点变换矩阵的逆，即T_o=T_inv, R_o=R_inv, S_o=S_inv
    o2w = T_inv @ R_inv @ S_inv
    w2o = S @ R @ T
    return o2w, w2o


def view_transform(r: Array, u: Array, l: Array, pos: Array) -> Tuple[Array, Array]:
    T, T_inv = translation(pos)
    R = np.eye(4)
    R[:3, :3] = np.stack([r, u, l], axis=0)
    R_inv = R.T
    M = R @ T
    M_inv = T_inv @ R_inv
    return M, M_inv


def projection(fov: float, alpha: float, n: float, f: float) -> Tuple[Array, Array]:
    fov_radians = fov * np.pi / 180
    tan = math.tan(fov_radians /2)
    P = np.array([
        [1 / (alpha * tan), 0., 0., 0.],
        [0., 1 / tan, 0., 0.],
        [0., 0., f / (f - n), f * n / (n - f)],
        [0., 0., 1., 0.]
    ])
    P_inv = np.array([
        [alpha * tan, 0., 0., 0.],
        [0., tan, 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., (n - f) / (f * n), 1 / n]
    ])
    return P, P_inv


def viewport(W: int, H: int) -> Tuple[Array, Array]:
    vp = np.array([
        [W / 2, 0., 0., W / 2],
        [0., - H / 2, 0., H / 2],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    vp_inv = np.array([
        [2 / W, 0., 0., -1.],
        [0., -2 / H, 0., 1.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ])
    return vp, vp_inv


def build_coordinate_system(z: Array) -> Tuple[Array, Array]:
    """
    给定`z`轴，建立坐标系，返回变换矩阵
    Args:
        z(Array): [3, N]，z轴
    Returns:
        matrix(Array): [N, 3, 3]，源坐标系到新坐标系的变换矩阵
        inv(Array): [N, 3, 3]，新坐标系到源坐标系的变换矩阵
    """
    z = normalize(z)
    x = np.array([[1.], [0.], [0.]])
    x = np.where(np.abs((x * z).sum(axis=0, keepdims=True)) + 1e-8 > 1.0,  np.array([[0.], [1.], [0.]]), x)
    y = normalize(np.cross(z, x, axis=0))
    x = normalize(np.cross(y, z, axis=0))
    inv = np.stack([x, y, z], axis=1).transpose((2, 0, 1))
    matrix = inv.transpose((0, 2, 1))
    return matrix, inv

