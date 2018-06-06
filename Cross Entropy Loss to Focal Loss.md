<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
#  DeepLearning - Focal loss

### 1.背景
**Object detection** 的算法主要可以分为两大类：

1. two-stage-detector
2. one-stage-detector

前者是类似`Faster RCNN`，`RFCN`这样需要`region proposal`的检测算法，这类算法可以达到很高的准确率，但是速度较慢。虽然可以通过减少 proposal 的数量或降低输入图像的分辨率等方式达到提速，但是速度并没有质的提升。后者是类似YOLO，SSD这样不需要region proposal，直接回归的检测算法，这类算法速度很快，但是准确率不如前者。不论如何，目前的目标检测算法都要 `trade-off` 在识别精度和实时性之间。鱼和熊掌都想吃？ [`RetinaNet`](https://arxiv.org/pdf/1708.02002.pdf)网络中，提出 **focal loss**  的出发点也是希望one-stage detector可以达到two-stage detector的准确率，同时不影响原有的速度。虽然说效果没有特别大的提升，精度和实时性都提高一点，值得研究。

又是 **Facebook AI Rearch **团队发表的！作者认为 one-stage detector 的准确率不如 two-stage detector 的原因是：**样本的类别不均衡导致的。**我们知道在object detection，一张图像可能生成成千上万的 candidate locations，但是其中只有很少一部分是包含object的，这就带来了类别不均衡（背景也算类别）。那么类别不均衡会带来什么后果呢？

1. 当训练样本中有很多背景单一，容易和目标区分，训练加再多张类似的图片对模型并没有太大用；
2. 负样本数量太大，占总的loss的大部分，而且多是容易分类的，因此使得模型的优化方向并不是我们所希望的那样。

之前也有一些算法来处理类别不均衡的问题，比如说 **OHEM**，其主思想概括：In OHEM each example is scored by its loss, non-maximum suppression (nms) is then applied, and a minibatch is constructed with the highest-loss examples。 **OHEM算法虽然增加了错分类样本的权重，但是OHEM算法忽略了容易分类的样本。**就是在forward的时候根据loss排序，然后选择loss最大的，也就是利用最困难的样本进行backward更新模型的参数。

针对类别不均衡问题，作者提出一种新的损失函数：**Focal loss**，这个损失函数是在标准交叉熵损失基础上修改得到的。这个函数可以通过减少易分类样本的权重，使得模型在训练时更专注于难分类的样本。



### 2. Focal loss

- **Cross Entropy**
  二分类为例，标准的交叉熵损失如下：
$$
  CrossEntropy=-\frac{1}{n}[y_ilog(p_i)+(1-y_i)log(1-log(p_i))] 
$$
  或者
$$
\begin{equation}  
  CE(p,y)=\left\{  
               \begin{array}{lr}  
              -log(p), &  if  \; y = 1\\  
               -log(1-p), &otherwise.\\    
               \end{array}  
  \right.  
  \end{equation}
$$
这里规定二分类 $y$ 的值是正 $1$ 或负 $1$，$p$ 是模型估计样本类别 $y=1$ 的概率，取值范围为 $0-1$。当真实 label 是 $1$，也就是 $y=1$ 时，假如某个样本 $x$ 预测为 $1$ 这个类的概率 $p=0.6$，那么损失就是$ -log(0.6)$，注意这个损失是大于等于  $0$ 的。如果 $p= 0.9$，那么损失就是 $-log(0.9)$，所以 $p=0.6$ 的损失要大于 $p=0.9$ 的损失，这很容易理解。为了方便，用 $p_t$ 代替 $p$，如下公式：
$$
\begin{equation}  
  p_t=\left\{  
               \begin{array}{lr}  
              p, &  if  \; y = 1\\  
               1-p, &otherwise\\    
               \end{array}  
  \right.  
  \end{equation}
$$
那么
$$
CE(p,y)=CE(p_t)=-log(p_t)
$$
接下来是一个最基本的对交叉熵的改进。

- **Balanced cross Entropy**
$$
CE(p_t)=-\alpha_t log(p_t)
$$
增加了一个系数 $\alpha_t$，跟 $p_t$ 的定义类似，当 $label=1$ 的时候，$\alpha_t=a$；当 $label=-1$ 的时候，$\alpha_t=1-a$，$a$ 的范围也是 $0$ 到 $1$。因此可以通过设定 $a$的值（一般而言假如 $1$ 这个类的样本数比 $-1$ 这个类的样本数多很多，那么 $a$ 会取 $0$ 到 $0.5$ 来增加 $-1$ 这个类的样本的权重）来控制正负样本对总的 $loss$ 的共享权重。这里当 $a=0.5$ 时就和标准交叉熵一样了（系数是个常数）。**显然 Balanced cross Entropy 的公式虽然可以控制正负样本的权重，但是没法控制容易分类和难分类样本的权重。**因此，Focal Loss 做了下面的改变。

- **Focal Loss**
$$
FL(p_t)=-{(1-p_t)}^{\gamma}log(p_t)
$$
这里的 $\gamma$  称作 focusing parameter，$\gamma>=0$。其中 $(1-p_t)^{\gamma}$  称为调制系数（modulating factor）。

这里介绍下focal loss的两个重要性质：
1. 当一样本被错分时，$p_t$ 很小，调制系数趋于 $1$，即相比原来的 loss 是没有大的改变。当 $p_t$ 趋于 $1$ 的时候（分类正确而且是易分类样本），调制系数趋于 $0$，也就是对于总的 loss的贡献很小，**而在 OHEM 这一部分的 loss 是被忽略的**。
2. 当 $\gamma=0$ 的时候，focal loss 就是传统的交叉熵损失，当 $\gamma$ 增加的时候，调制系数也会增加。 

focal loss 的这两个重要性质，其实就是用一个合适的函数去**度量难分类和易分类样本对总的损失的贡献**。

还没完！结合 Balanced cross Entropy 的公式，对 focal loss 作进一步改进：
$$
FL(p_t)=-{\alpha_t}{(1-p_t)}^{\gamma}log(p_t)
$$

显而易见，这样**既能调整正负样本的权重，又能控制难易分类样本的权重**。
