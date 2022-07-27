### eigenface特征脸

PCA过程：

1. 把人脸图片张成一个列向量，N张图片表示为$(X_1,X
   _2,X_3,...,X_N)$，其中$X_i$是一个n维（$n=h*w$）的列列向量，并进行标准化 $X=X-\hat{X}$（均值$\hat{X}=\frac{\sum\limits_{i=1}^NX_i}{N}$）。
   
   $A=(\begin{matrix}X_1&X_2&X_3&...&X_N\end{matrix})_{n*N}$
   
2. 构造方阵$C=(AA^T)_{n*n}$求方阵C的特征值及特征向量

3. 如何求方阵C的特征向量呢，如果直接求的话计算量太大万维以上级别，论文中提供了一种巧妙的方法。

   由于$AA^T$维度太大，但$A^TA$维度却只有$N(<<n)$，因此
   $$
   \begin{aligned}
   A^TAV_i&=\lambda_iV_i\\
   AA^T(AV_i)&=\lambda_i(AV_i)\\
   AA^TU_i&=\lambda_iU_i
   \end{aligned}
   $$
   $V_i$为$A^TA$的$\lambda_i$特征值对应的特征向量

   通过这个转换，就把求$AA^T$的特征向量变成求$A^TA$的特征向量，计算量大大减少

   ![image-20220708180702658](assets/image-20220708180702658.png)

4. 论文中提到，由特征值$\lambda_k$取得最大值时对应的特征向量$U_k$作为特征脸，由此公式形成损失函数
   $$
   \lambda_k=\frac{1}{M}\sum\limits_{i=1}^M(U_k^TX_i)^2 \begin{matrix}&\text{is}\text{ a }\text{maximum.}\end{matrix}
   $$
   为何可以这样算，为什么是求最大值呢？如图

   我们把$X_i$看成是特征脸空间上的一个点（注意此时的$X_i$已经被处理为$n$维的列向量），把特征脸集$U$看成这个特征空间上的一条线，我们要把$X_i$投影到特征脸集$U$上来，当然是误差越小越好，而误差可以看出是点到线的距离，那么就是点到线的距离越小越好，这个问题又可以转化为与距离垂直的$U$上的投影越大越好，因为点到原点长度固定，当点到线距离越小那么另一条直角边也就是越大越好，最终问题转化到投影（也就是这个$\lambda$）越大越好。（$X_i^TU$和$U^TX_i$结果一样都是点积处理后的一个数）

   ![image-20220708164418748](.\assets\image-20220708164418748.png)

5. 得到一组特征值、特征向量带入特征公式得到，$\lambda_i$从大到小排
   $$
   \begin{aligned}
   AA^T
   &=(\begin{matrix}U_1&U_2&U_3&...&U_n\end{matrix})\left(\begin{matrix}\lambda_1&&&&\\&\lambda_2&&&\\&&\ddots&\\
   &&&\lambda_n\end{matrix}\right)\\
   &=\lambda_1U_1U_1^T+\lambda_2U_2U_2^T+\cdots+\lambda_nU_nU_n^T
   \end{aligned}
   $$
   又方阵C的秩等于400，所以特征值最多取到400，即
   $$
   AA^T=\lambda_1U_1U_1^T+\lambda_2U_2U_2^T+\cdots+\lambda_kU_kU_k^T+\cdots(可舍去)
   $$
   式中的$\lambda_iU_iU_i^T$即为一个输入图片对应的特征脸的一个表示

   每个人脸可以认为是特征脸集的加权和，即又把人脸映射到特征脸上（k个特征脸，一个特征脸对应一个权重）
   $$
   （人脸）X=\sum\limits_{i=1}^{400}W_iU_i（特征脸）
   $$
   可以用小于$K（如取100,舍弃100后的特征值）$特征脸的子空间来近似代表特征脸。（因为特征值按大到小排序，前面100个已经占据绝大多数特征）

   $U_i$和$U_i^T$是标准正交基，求权重$W_i$可以通过对人脸列向量$X$左乘行向量$U_j^T$
   $$
   \begin{aligned}
   U_j^TX
   &=\sum\limits_{i=1}^{k}W_iU_j^TU_i\\
   &=W_j
   \end{aligned}
   $$
   其中$U_j^TU_i=\begin{cases}1,&i=j\\0,&i\neq j\end{cases}$，得到关于$j$的权重，所有权重组合起来形成特征脸权重$W_i=(\begin{matrix}W_1 & W_2 & W_3 & ...&W_k\end{matrix})$

   ![image-20220708153043195](.\assets/image-20220708153043195.png)

6. 人脸描述转移到用一组权重W表示，最终达到降维的效果，把$n^2$（像素数）维降到400（特征脸权重数）

![image-20220707223700609](.\assets\image-20220707223700609.png)





