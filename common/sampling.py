from device import np, Array, random_float32


def sample_disk(R: float, n: int) -> Array:
    """
    在圆形内均匀采样
    Args:
        R(float): 半径
        n(int): 采样点数量
    Returns:
        sample(Array): [2, n]
    """
    e1 = random_float32(n)
    e2 = random_float32(n)
    r = R * np.sqrt(e1)
    theta = 2 * np.pi * e2
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=0)


def sample_hemisphere(n: int) -> Array:
    """
    单位半球面均匀采样
    Args:
        n(int): 采样点数量
    Returns:
        sample(Array): [3, n]
    """
    e1 = random_float32(n)
    e2 = random_float32(n)
    cos_theta = e1
    sin_theta = np.sqrt(1 - e1 ** 2)
    phi = 2 * np.pi * e2
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return np.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=0)


def sample_sphere(n: int) -> Array:
    """
    单位球面采样
    Args:
        n(int): 采样点数量
    Returns:
        sample(Array): [3, n]
    """
    e1 = random_float32(n)
    e2 = random_float32(n)
    cos_theta = 1 - 2 * e1
    sin_theta = np.sqrt(np.clip(1 - cos_theta ** 2, a_min=0.0, a_max=1.0))
    phi = 2 * np.pi * e2
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return np.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], axis=0)


def sample_triangle(p0: Array, p1: Array, p2: Array, n: int) -> Array:
    """
    三角形内均匀采样
    Args:
        p0(Array): [3, 1]，三角形顶点
        p1(Array): [3, 1]，三角形顶点
        p2(Array): [3, 1]，三角形顶点
        n(int): 采样点数量
    Returns:
        sample(Array): [3, n]
    """
    e1 = random_float32(n)
    e2 = random_float32(n)
    sqrt_e1 = np.sqrt(e1)
    u = 1 - sqrt_e1
    v = e2 * sqrt_e1
    return (1 - u - v) * p0 + u * p1 + v * p2