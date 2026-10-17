# 三角形均匀采样一般情形
设 $\Delta P_0P_1P_2$ 顶点坐标为 $P_0:(x_0,y_0,z_0),P_1:(x_1,y_1,z_1),P_2:(x_2,y_2,z_2)$，三角形内一点 $P:(x,y,z)=u\cdot P_0+v\cdot P_1+w\cdot P_2, u+v+w=1, u,v,w\geqslant 0$，即
$$
\begin{equation}
    \begin{pmatrix}
        x_0 & x_1 & x_2 \\
        y_0 & y_1 & y_2 \\
        z_0 & z_1 & z_2 \\
        1 & 1 & 1
    \end{pmatrix}
    \begin{pmatrix}
        u \\ v \\ w
    \end{pmatrix}=\begin{pmatrix}
        x \\ y \\ z \\ 1
    \end{pmatrix}
\end{equation}
$$
由于 $P_0,P_1,P_2,P$ 共面，故向量组 $\overrightarrow{PP_0},\overrightarrow{PP_1},\overrightarrow{PP_2}$ 的秩为2，即矩阵
$$
\begin{equation}
    \begin{pmatrix}
        x_0-x & x_1-x & x_2-x \\
        y_0-y & y_1-y & y_2-y \\
        z_0-z & z_1-z & z_2-z
    \end{pmatrix}
\end{equation}
$$
的秩为2，线性方程组 $(1)$ 增广矩阵的行列式为
$$
\begin{equation}
    \begin{split}
        \begin{vmatrix}
            x_0 & x_1 & x_2 & x \\
            y_0 & y_1 & y_2 & y \\
            z_0 & z_1 & z_2 & z \\
            1 & 1 & 1 & 1
        \end{vmatrix}&=\begin{vmatrix}
            x_0-x & x_1-x & x_2-x & x \\
            y_0-y & y_1-y & y_2-y & y \\
            z_0-z & z_1-z & z_2-z & z \\
            0 & 0 & 0 & 1
        \end{vmatrix}\\
        &=\begin{vmatrix}
            x_0-x & x_1-x & x_2-x \\
            y_0-y & y_1-y & y_2-y \\
            z_0-z & z_1-z & z_2-z
        \end{vmatrix}\\
        &=0
    \end{split}
\end{equation}
$$
即增广矩阵的秩不大于3   
取系数矩阵的一个三阶子式
$$
\begin{equation}
    \begin{split}
        \begin{vmatrix}
            x_0 & x_1 & x_2 \\
            y_0 & y_1 & y_2 \\
            1 & 1 & 1
        \end{vmatrix}&=\begin{vmatrix}
            x_0-x_2 & x_1-x_2 & x_2 \\
            y_0-y_2 & y_1-y_2 & y_2 \\
            0 & 0 & 1
        \end{vmatrix}\\
        &=\begin{vmatrix}
            x_0-x_2 & x_1-x_2 \\
            y_0-y_2 & y_1-y_2 \\
        \end{vmatrix}\\
    \end{split}
\end{equation}
$$
由于 $\overrightarrow{P_2P_0},\overrightarrow{P_2P_1}$ 不共线，故 $(4)$ 存在不为0的二阶子式，因此线性方程组
$$
\begin{equation}
    \begin{pmatrix}
        x_0 & x_1 & x_2 \\
        y_0 & y_1 & y_2 \\
        1 & 1 & 1
    \end{pmatrix}
    \begin{pmatrix}
        u \\ v \\ w
    \end{pmatrix}=\begin{pmatrix}
        x \\ y \\ 1
    \end{pmatrix}
\end{equation}
$$
存在唯一解，记
$$
\begin{align*}
    D&=\begin{pmatrix}
        x_0 & x_1 & x_2 \\
        y_0 & y_1 & y_2 \\
        1 & 1 & 1
    \end{pmatrix} \\
    D_0&=\begin{pmatrix}
        x & x_1 & x_2 \\
        y & y_1 & y_2 \\
        1 & 1 & 1
    \end{pmatrix} \\
    D_1&=\begin{pmatrix}
        x_0 & x & x_2 \\
        y_0 & y & y_2 \\
        1 & 1 & 1
    \end{pmatrix}
\end{align*}
$$
由 Cramer 法则
$$
\begin{equation}
    \begin{split}
        u&=\frac{|D_0|}{|D|} \\
        v&=\frac{|D_1|}{|D|}
    \end{split}
\end{equation}
$$
雅可比行列式为
$$
\begin{equation}
    \begin{split}
        \frac{\partial(u,v)}{\partial(x,y)}&=\begin{vmatrix}
            \cfrac{y_1-y_2}{|D|} & -\cfrac{x_1-x_2}{|D|} \\
            -\cfrac{y_0-y_2}{|D|} & \cfrac{x_0-x_2}{|D|}
        \end{vmatrix} \\
        &=\frac{1}{|D|^2}\cdot\begin{vmatrix}
            y_1-y_2 & -(x_1-x_2) \\
            -(y_0-y_2) & x_0-x_2
        \end{vmatrix} \\
        &=\frac{1}{|D|^2}\cdot\begin{vmatrix}\begin{pmatrix}
            x_0-x_2 & y_0-y_2 \\
            x_1-x_2 & y_1-y_2
        \end{pmatrix}^*\end{vmatrix} \\
        &=\frac{1}{|D|^2}\cdot\begin{vmatrix}
            x_0-x_2 & y_0-y_2 \\
            x_1-x_2 & y_1-y_2
        \end{vmatrix} \\
        &=\frac{1}{|D|^2}\cdot\begin{vmatrix}
            x_0-x_2 & y_0-y_2 & 0 \\
            x_1-x_2 & y_1-y_2 & 0 \\
            x_2 & y_2 & 1
        \end{vmatrix} \\
        &=\frac{1}{|D|}
    \end{split}
\end{equation}
$$
不失一般性，设 $\Delta P_0P_1P_2$ 所在平面不与 $xOy$ 平面垂直，平面方程为 $z=ax+by+c$，点 $P_0,P_1,P_2$ 在 $xOy$ 平面的投影分别为 $Q_0,Q_1,Q_2$，则 $\Delta P_0P_1P_2$ 内均匀采样的随机变量 $(X,Y,Z)$ 满足
$$
\begin{align}
    f(x,y)&=\begin{cases}
        \cfrac{2}{
            \begin{Vmatrix}
                \overrightarrow{Q_2Q_0}\times\overrightarrow{Q_2Q_1}
            \end{Vmatrix}^2
        }, & (x,y,0)^T=u\cdot\overrightarrow{OQ_0}+v\cdot\overrightarrow{OQ_1}+w\cdot\overrightarrow{OQ_2} \\
        0
    \end{cases} \\
    Z&=aX+bY+c
\end{align}
$$
化简得
$$
\begin{align*}
    \begin{Vmatrix}
        \overrightarrow{Q_2Q_0}\times\overrightarrow{Q_2Q_1}
    \end{Vmatrix}^2&=|(x_0-x_2)(y_1-y_2)-(x_1-x_2)(y_0-y_2)| \\
    &=\begin{vmatrix}
        |D|
    \end{vmatrix}
\end{align*}
$$
从而 $(u,v)$ 的概率密度为
$$
\begin{align}
    f(u,v)&=\begin{cases}
        2, & u+v\leqslant 1,u,v\geqslant 0 \\
        0
    \end{cases}
\end{align}
$$
之后按 <a href="https://www.bilibili.com/video/BV164Gu6fEs1">第12讲</a> 的方法即可推广到一般三角形的均匀采样