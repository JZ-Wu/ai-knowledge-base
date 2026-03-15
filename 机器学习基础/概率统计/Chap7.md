# 7.参数估计

这部分内容中，我们的研究对象都将是总体 $X$。对于总体 $X$，我们可能求的是未知的分布。也可能是已知分布形式的前提下，分布中含有未知参数。我们将要对总体 $X$ 进行估计，来研究总体，进行统计推断。

为了实现估计，我们采取从总体中抽样的方法（独立重复试验），得到样本，相互独立且与总体同分布。根据不同需要，我们构造不同的统计量 $g(X_1,X_2,\cdots,X_n)$

## 点估计与矩估计

### 点估计

问题：

设总体 $X$ 的分布函数（概率密度，分布律）的形式已知，但它的一个或多个参数未知，借助于总体 $X$ 的一个样本来估计总体未知参数的值的问题，称为参数的点估计问题。

已知：总体 $X$ 的分布函数的 $F(x;\theta)$ 的形式；

未知：$\theta$ 待估参数。

利用：$X_1,X_2,\cdots,X_n$ 是 $X$ 的一个样本，$x_1,x_2,\cdots,x_n$ 是相应的一个样本值。

**如何解决？**

构造一个适当的统计量 $\hat \theta (X_1,X_2,\cdots,X_n)$，用它的观察值 $\hat \theta (x_1,x_2,\cdots,x_n)$ 作为未知参数 $\theta$ 的近似值。

称 $\hat \theta(X_1,X_2,\cdots,X_n)$ 为 $\theta$ 的估计量。

称 $\hat \theta(x_1,x_2,\cdots,x_n)$ 为 $\theta$ 的估计值。

> 对不同样本值，估计值一般不同。

### 矩估计

**理论依据：** 样本的 $k$ 阶矩依概率收敛于总体的 $k$ 阶矩，即：

$$
A_k=\frac1n\sum_{i=1}^n X_i^k \rightarrow^P E(X^k) 
$$

也就是说，当 $n \to \infty$ 时，可以用 $A_k$ 近似估计 $E(X^k)$

**求解步骤**：

设总体 $X$ 的分布中含有 $m$ 个未知参数 $\theta_1,\theta_2,\cdots,\theta_m$

* 求总体的各阶矩 $E(X^k)$ $(k=1,2,\cdots,m)$

* 令样本的各阶矩等于总体的各阶矩，得到含 $m$ 个未知参数 $\theta_1,\theta_2,\cdots,\theta_m$ 的方程。

$$
\begin{cases}
\frac1n \sum_{i=1}^nX_i=E(X) \\
\frac1n \sum_{i=1}^nX_i^2=E(X^2) \\
\vdots \\
\frac1n \sum_{i=1}^nX_i^m=E(X^m)
\end{cases}
$$

> 总体中有几个未知参数，就建立几个方程

* 解上述方程，所求得的解 $\hat \theta_k(X_1,X_2,\cdots,X_n)$ 称为未知参数 $\theta_k$ 的矩估计量，简称矩估计。

> 实际情况中我们得到的都是样本值，将样本值代入样本进行计算得到的 $\hat \theta$ 使我们的一次估计情况。每次估计值都可能不同。

## 最大似然估计

设总体 $X$ 为离散型，分布律已知，但分布中含有 $m$ 个未知参数 $\theta_1,\theta_2,\cdots,\theta_n$ ， $X_1,X_2,\cdots,X_n$ 是 $X$ 的一个样本，$x_1,x_2,\cdots,x_n$ 是相应的一个样本值。

易知 $X_1,X_2,\cdots,X_n$ 取到观察值 $x_1,x_2,\cdots,x_n$ 的概率，即事件 $p\{X_1=x_1,X_2=x_2,\cdots,X_n=x_n\}$ 发生的概率为：

$$
L(x_1,x_2,\cdots,x_n;\theta_1,\theta_2,\cdots,\theta_m)=\prod_{i=1}^n P\{X=x_i\}
$$

这一概率随 $\theta_1,\theta_2,\cdots,\theta_m$ 的取值而变化，它是 $m$ 个未知参数 $\theta_1,\theta_2,\cdots,\theta_n$ 的函数，称其为样本的似然函数，需要注意的是 $x_1,x_2,\cdots,x_n$ 是已知的样本值。

最大似然函数的思想：已经取到样本值 $x_1,x_2,\cdots,x_n$ 了，这就说明取到这一样本值的概率 $L(x_1,x_2,\cdots,x_n;\theta_1,\theta_2,\cdots,\theta_m)$ 比较大。因此，可以固定样本观察值 $x_1,x_2,\cdots,x_n$，挑选使似然函数

$$
L(x_1,x_2,\cdots,x_n;\theta_1,\theta_2,\cdots,\theta_m)=\prod_{i=1}^n P\{X=x_i\}
$$

达到最大值的参数值

$$
\hat \theta_i (x_1,x_2,\cdots,x_n) (i=1,2,\cdots,m)
$$

用这种思想求出的参数值称为 $\theta_1,\theta_2,\cdots,\theta_m$ 的最大似然估计值。

相应的统计量 $\hat \theta_i (x_1,x_2,\cdots,x_n) (i=1,2,\cdots,m)$ 称为参数的最大似然估计量。

---

最大似然估计法的解题步骤可以概括如下：

**第1步：确定概率模型**

- 明确数据是如何生成的，是什么分布。比如是正态分布、二项分布、泊松分布等。

**第2步：写出似然函数**

- 假设有一组样本数据 $X = (x_1, x_2, \ldots, x_n)$，概率模型中的参数记为 $\theta$。
- 写出样本数据的联合概率密度或质量函数，即似然函数 $L(\theta)$。如果样本是独立同分布的，那么 $L(\theta) = \prod_{i=1}^{n} f(x_i|\theta)$，其中 $f(x_i|\theta)$ 是第 $i$ 个样本的概率密度或质量函数。

**第3步：对似然函数取对数转化为对数似然函数**

- 对数似然函数定义为 $l(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log f(x_i|\theta)$。这样做的好处是将乘法转化为加法，简化了求导和计算。

**第4步：对对数似然函数求导**

- 分别对参数 $\theta$ 求导，得到一阶导数，即梯度 $\nabla_\theta l(\theta)$。

**第5步：求解似然方程**

- 将一阶导数设为 $0$，解得的方程称为似然方程 $\nabla_\theta l(\theta) = 0$。
- 解这个方程组可能得到一个或多个参数的解。这可能涉及解析方法或者数值求解。

**第6步：找到最大值**

- 验证所得解确实为最大值。这可以通过检查二阶导数是否为负（即二阶导数矩阵，也称为 Hessian 矩阵，是负定的）。
- 在多参数情况下，确保所得点是局部最大值而不是鞍点。

**第7步：评估和解释结果**

- 评估最大似然估计得到的参数值是否合理。
- 在必要时，可以通过计算置信区间或进行假设检验来进一步分析参数。

**示例**

假设我们有一个样本 $X = (x_1, x_2, \ldots, x_n)$，我们假设样本来自于均值为 $\mu$，方差为 $\sigma^2$ 的正态分布。

1. **概率模型**：正态分布 $N(\mu, \sigma^2)$。
   
2. **似然函数**：
   $L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(x_i-\mu)^2}{2\sigma^2} \right)$。

3. **对数似然函数**：
   $l(\mu, \sigma^2) = \sum_{i=1}^{n} \left[ -\frac{1}{2} \log(2\pi\sigma^2) -\frac{(x_i-\mu)^2}{2\sigma^2} \right]$。

4. **求导数**：
   $\frac{\partial l}{\partial \mu} = \sum_{i=1}^{n} \frac{x_i - \mu}{\sigma^2}$；
   $\frac{\partial l}{\partial \sigma^2} = \sum_{i=1}^{n} \left[ -\frac{1}{2\sigma^2} +\frac{(x_i - \mu)^2}{2(\sigma^2)^2} \right]$。

5. **解似然方程**：
   令导数为零，解得 $\hat{\mu} = \bar{x}$（样本均值），$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2$（样本方差）。

6. **验证**：
   对二阶导数检查确保得到的是最大值。

通过这个过程，我们可以得到参数 $\mu$ 和 $\sigma^2$ 的最大似然估计。

## 估计量的评选标准

### 无偏性

**定义**：设 $\hat \theta=\theta(X_1,X_2,\cdots,X _n)$ 是 $\theta$ 的估计量，若 $E(\hat \theta)=\theta$，则称 $\hat \theta=\theta(X_1,X_2,\cdots,X_n)$ 是 $\theta$ 的无偏估计。

### 有效性

**定义**：设 $\hat \theta_1$，$\hat \theta_2$ 均为 $\theta$ 的无偏估计，若 $D(\hat \theta_1) \leq D(\hat \theta_2)$，则称 $\hat \theta_1$ 较 $\hat \theta_2$ 有效。

> $E(\hat \theta_1)=E(\hat \theta_2)=\theta$，比较 $D(\hat \theta_1),D(\hat \theta_2)$ 大小。

### 相合性

无偏性和有效性都是在样本容量 $n$ 固定的前提下讨论，我们希望样本容量越大，$\hat \theta$ 与 $\theta$ 更接近。

**定义**：设 $\hat \theta=\theta(X_1,X_2,\cdots,X_n)$ 是 $\theta$ 的估计量，若对任意的 $\epsilon>0$，有 $\lim_{n \to \infty}P\{|\hat \theta - \theta |<\epsilon \}=1$，即 $\hat \theta=\theta(X_1,X_2,\cdots,X_n)$ 依概率收敛于 $\theta$，则称 $\hat \theta=\theta(X_1,X_2,\cdots,X_n)$ 是 $\theta$ 的相合估计量。

> 相合性是对估计量的一个基本要求。

## 置信区间

在参数估计中，我们有点估计和区间估计。对于点估计，我们得到的是未知参数的一个近似值。我们有些时候会认为这个值有点粗糙，并且不能反映出估计的精确程度。

于是，我们尝试进行区间估计。区间估计的好处是给出参数值的一个范围，并给出此范围内包含参数 $\theta$ 真值的可信程度。

例如，对于第二天的气温估计，我们认为 $80\%$ 的概率为 $(27,30)$ 度，这里 $80\%$ 就是可信程度，$(27,30)$ 就是置信区间。

一般情况下，我们使用区间长度来刻划精确度。在就可信程度不变的条件下，区间越短，精确度越高。也就是说，在精确度要求确定的情况下，我们会选择区间最短的那段区间作为我们的置信区间。

下面我们对置信区间进行更为精确的定义：

---

**定义**：设总体 $X$ 的分布函数为 $F(x;\theta)$，含有一个未知参数 $\theta$，$\theta \in \Theta$，（$\Theta$ 是 $\theta$ 的取值范围）

> 我们通过抽样确定总体 $X$ 的分布函数，因此 $\theta$ 会存在取值范围，参考 MLE

对应给定值 $\alpha (0<\alpha<1)$，若由来自 $X$ 的样本 $X_1,X_2,\cdots,X_n$ 确定的两个统计量

$$
\underline \theta=\theta(X_1,X_2,\cdots,X_n), \ \bar \theta =\theta(X_1,X_2,\cdots,X_n) \ (\underline \theta< \bar \theta)
$$

对于任意的 $\theta \in \Theta$ 满足：

$$
P\{\underline \theta(X_1,X_2,\cdots,X_n)<\theta<\ \bar \theta(X_1,X_2,\cdots,X_n)\}\geq 1-\alpha
$$

也就是 $\theta$ 落在该区间的概率大于 $1-\alpha$，则称随机区间 $(\underline \theta,\bar \theta)$ 是 $\theta$ 的置信水平为 $1-\alpha$ 的置信区间，$\underline \theta$ 为置信区间的置信下限，$\bar \theta$ 为置信区间的置信上限，$1-\alpha$ 为置信水平。

需要注意的是，$\alpha$ 的取值一般都很小。

---