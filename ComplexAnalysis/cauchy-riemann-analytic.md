---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
---



We will adopt the convention, popular in engineering of substituing the "imaginary" number $i$ ( $i^2=-1$) with $j$ instead because in our linear algebra work we have been using $I$ as the identity matrix and $J$ as the matrix which does a quarter turn counter clockwise.
$$
I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$
$$
J = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}
$$
This allowed us to make note the following relationship between complex numbers and a space with identical properties (an "isomorphism") via:
$$
z = x I + y J = \begin{bmatrix} x & -y \\ y & x \end{bmatrix}
$$
$$
x I + y J \leftrightarrow x + j \, y 
$$
for $x,y \in \mathbb{R}$. Thus in general a complex number can be represented by an anti-symmetric 2x2 matrix.


$$
z = x + j y \leftrightarrow \begin{bmatrix} x \\ y \end{bmatrix}
$$
$$
f(z) = u(x,y) + j v(x,y)
$$
$$
f(z_0 + h) = f(z_0) + Df \big|_{z0} [h] 
$$
$$
[Df]_E = D u(x,y) + j D v(x,y) = \partial_x u \, dx + \partial_y u \, dy + j (\partial_x v \, dx + \partial_y v \, dy )
$$

$$
\begin{bmatrix}
\frac{\partial u}{\partial x}  & \frac{\partial u}{\partial y}\\
\frac{\partial v}{\partial x} & \frac{\partial  v}{\partial y}
\end{bmatrix}
$$


Let's assume these derivatives exist for the moment. Then for $Df|_{z0}$ to act like a complex number it needs to be anti-symmetric and its two diagonal values need to be equal. This gives us the Cauchy-Rieman conditions:
$$
\frac{\partial u}{\partial x}  = \frac{\partial  v}{\partial y} , \,\,
\frac{\partial v}{\partial x} = - \frac{\partial  u}{\partial y}
$$



