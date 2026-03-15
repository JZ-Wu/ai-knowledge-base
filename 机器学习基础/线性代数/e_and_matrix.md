# e的矩阵指数与微分方程

## 定义

设$A$是一个$n$阶方阵,$I$是$n$阶单位矩阵,则矩阵指数函数$e^{At}$定义为:

$$
e^{At} = (At)^0 + \frac{(At)^1}{1!} + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots
\tag{1}
$$

也可以表示为：

$$
e^{At} = \left(\begin{bmatrix} a & b \\ c & d \end{bmatrix} \cdot t\right)^0 + \frac{\left(\begin{bmatrix} a & b \\ c & d \end{bmatrix} \cdot t\right)^1}{1!} + \frac{\left(\begin{bmatrix} a & b \\ c & d \end{bmatrix} \cdot t\right)^2}{2!} + \frac{\left(\begin{bmatrix} a & b \\ c & d \end{bmatrix} \cdot t\right)^3}{3!} + \cdots \tag{2}
$$

## 有什么用？

从微分方程组的角度来了解 $e$ 的矩阵指数

$$
\begin{cases}
\frac{dx}{dt} = ax(t) + by(t)
\\
\frac{dy}{dt} = cx(t) + dy(t)
\end{cases}
\tag{3}
$$


可以将上述方程组用矩阵形式表示如下：

$$
 \begin{bmatrix}
x'(t) \\
y'(t)
\end{bmatrix} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix} \begin{bmatrix}
x(t) \\
y(t)
\end{bmatrix}
\tag{4}
$$

从线性变换的几何角度理解这个式子，点$(x,y)$的变化率等于位置向量根据矩阵线性变换后得到的速度向量方向和大小。

把矩阵和方程组用微分简写表示后，得到

$$
\frac{d}{dt}\vec{v}(t)=M\vec{v}(t)
\tag{5}
$$

接下来与一个基本的微分方程作对比：

$$
\frac{d}{dt}x(t)=r \cdot x(t)
\tag{6}
$$

我们都知道，这个微分方程的解为

$$
x(t)=Ke^{rt}
\tag{7}
$$

其中$K=x(0)$的值

对比 $(5)$ 式和 $(6)$ 式，我们于是希望 $(5)$式的解的结果也可以用如下的形式来表示：

$$
\begin{bmatrix}
x(t) \\
y(t)
\end{bmatrix} = e^{\begin{bmatrix}
a & b \\
c & d
\end{bmatrix} t} \begin{bmatrix}
x(0) \\
y(0)
\end{bmatrix}
\tag{8}
$$

**矩阵指数的定义很大程度上就是为了保证上面式子的正确性。**

## 怎么算？

$$
e^{At} = S(e^{\Lambda t})S^{-1}
\tag{9}
$$

其中:

- $A$ 是一个方阵
- $t$ 是时间变量  
- $S$ 是一个非奇异矩阵
- $e^{\Lambda t}$ 表示对角矩阵,其对角线元素为 $e^{\lambda_i t}$,$\lambda_i$ 是矩阵A的特征值
- $S$ 矩阵由矩阵A的特征向量构成

换句话说，矩阵指数函数$e^{At}$可以用矩阵$A$的特征值和特征向量表示出来。其中$S$矩阵由$A$的特征向量构成。

[查询笔记](/math/linear_algebra/MIT-Linear-Algebra-Notes/线性代数23.pdf)

## 题外话：关于薛定谔方程

$$
\frac{d}{dt}\vec{v}(t)=M\vec{v}(t)
\tag{1}
$$

将 $(1)$ 式对比薛定谔方程：

$$
i\hbar \frac{\partial \Psi}{\partial t} = H \Psi
\tag{2}
$$

> * $\psi$ 代表一个具体的描述系统状态的向量，封装一个系统内你关心的所有信息，例如粒子的位置和动量

$$
\frac{\partial \Psi}{\partial t} =(\frac{1}{i\hbar} H) \Psi
\tag{3}
$$

这个方程告诉我们状态向量的变化率等于一个矩阵乘上它自己。

薛定谔方程部分表示一系列的旋转。有待深入理解，这里不多写，内容可能也有些问题。

> 本文部分参考3b1b的视频。