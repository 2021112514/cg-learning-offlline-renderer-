# 坐标变换导致的曲面采样PDF变化
（因笔者没有系统地学过微分几何，该推导借助了DeepSeek）  
设局部坐标为 $x_l$，其PDF为 $p_l(x_l)$，世界坐标 $x_w=Wx_l$  
曲面 $x(u,v)$ 的面积元是
$$
\begin{align}
    dA=\begin{Vmatrix}\cfrac{\partial x}{\partial u}\times\cfrac{\partial x}{\partial v}\end{Vmatrix}_2dudv
\end{align}
$$
理解：平面上的小矩形 $(du,dv)$ 在曲面上变成了平行四边形 $(x_udu,x_vdv)$ ，其面积是叉乘的模  
将 $x_w$ 带入 $(1)$：
$$
\begin{equation}
    \begin{split}
        dA_w&=\begin{Vmatrix}\cfrac{\partial x_w}{\partial u}\times\cfrac{\partial x_w}{\partial v}\end{Vmatrix}_2dudv \\
        &=\begin{Vmatrix}(\cfrac{\partial x_w}{\partial x_l}\cfrac{\partial x_l}{\partial u})\times(\cfrac{\partial x_w}{\partial x_l}\cfrac{\partial x_l}{\partial v})\end{Vmatrix}_2dudv \\
        &=\begin{Vmatrix}(W\cfrac{\partial x_l}{\partial u})\times(W\cfrac{\partial x_l}{\partial v})\end{Vmatrix}_2dudv
    \end{split}
\end{equation}
$$
利用叉积恒等式
$$
\begin{align}
    (Wa)\times(Wb)=\det(W)W^{-T}(a\times b)
\end{align}
$$
有
$$
\begin{equation}
    \begin{split}
        dA_w&=|\det{W}|\cdot\begin{Vmatrix}W^{-T}\left(\cfrac{\partial x_l}{\partial u}\times\cfrac{\partial x_l}{\partial v}\right)\end{Vmatrix}_2dudv \\
        &=|\det{W}|\cdot\begin{Vmatrix}W^{-T}n_l\end{Vmatrix}_2\cdot\begin{Vmatrix}\cfrac{\partial x_l}{\partial u}\times\cfrac{\partial x_l}{\partial v}\end{Vmatrix}_2dudv \\
        &=|\det{W}|\cdot\begin{Vmatrix}W^{-T}n_l\end{Vmatrix}_2dA_l
    \end{split}
\end{equation}
$$
其中 $n_l$ 是局部坐标系下的单位法向量
因此世界坐标系下的PDF满足
$$
\begin{align}
    p_w(x_w)=\frac{p_l(x_l)}{|\det{W}|\cdot\begin{Vmatrix}W^{-T}n_l\end{Vmatrix}}
\end{align}
$$