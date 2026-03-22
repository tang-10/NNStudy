import torch
import torch.nn.functional as F


def focal_loss(inputs, targets, alpha=0.75, gamma=2):
    # 获取每一个二分类的概率。注意这里使用的是，多个二分类实现的多分类
    p = torch.sigmoid(inputs)
    # 多个二分类分别求出损失
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    """
	pt 的意思是什么呢？ (p是模型输出的概率)
	若targets=0.9 p=0.9 --> 表示模型基本是预测对了 
	p * targets=0.81 | (1 - p) * (1 - targets)=0.01 --> pt=0.82 --> 可以看到pt值较大
	若targets=0.9 p=0.1 --> 表示模型基本是预测错了
	p * targets=0.09 | (1 - p) * (1 - targets)=0.09 --> pt=0.18 --> 可以看到pt值较小
	若targets=0.1 p=0.9 --> 表示模型基本是预测错了
	p * targets=0.09 | (1 - p) * (1 - targets)=0.09 --> pt=0.18 --> 可以看到pt值较小
	若targets=0.1 p=0.1 --> 表示模型基本是预测对了
	p * targets=0.01 | (1 - p) * (1 - targets)=0.81 --> pt=0.82 --> 可以看到pt值较大

	所以pt值表示模型预测正确的一个程度值
	"""
    p_t = p * targets + (1 - p) * (1 - targets)

    """
	**********************************
	解决问题1: 控制容易分类和难分类样本的权重
	**********************************
	pt 越大 表示模型预测效果越好  更新程度不需要那么大( 1-pt 值就小)
	pt 越小 表示模型预测效果越差  更新程度需要大一些( 1-pt 值越大)
	p_t 越大越容易预测，p_t越小越不容易预测

	对于难以预测的目标，意味着模型的预测值与真实值之间的差距较大, 需要加大惩罚。
	"""
    loss = ce_loss * ((1 - p_t) ** gamma)

    """
	举例说明解决正负样本平衡问题:
	这里是前面的代码
	inputs = torch.tensor([-2, 2, 1.5])
	targets = torch.tensor([0., 1., 0.])
	p = F.sigmoid(inputs)
	# tensor([0.1192, 0.8808, 0.8176])
	print(p)

	ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
	# tensor([0.1269, 0.1269, 1.7014])
	print(ce_loss)

	p_t = p * targets + (1-p) * (1-targets)
	# tensor([0.8808, 0.8808, 0.1824])

	loss = ce_loss * (1-p_t) ** gamma
	# 分对的概率越高 1-pt就会越小
	# tensor([0.0018, 0.0018, 1.1373])
	print(loss)

	"""
    # tensor([0.2500, 0.7500, 0.2500])
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    """
	********************
	解决问题2:  正负样本平衡
	********************
	"""
    # 有target我们可以得到，当前类别是第2个类别，这样其他都是负样本，减少权重
    # tensor([0.2500, 0.7500, 0.2500])
    print(alpha_t)
    # 之前的损失
    # tensor([0.0018, 0.0018, 1.1373])
    # 当前的损失 之前的负样本都特别大，这里对负样本进行降权
    # tensor([0.0005, 0.0014, 0.2843])
    print(loss)
    return loss.mean()


if __name__ == '__main__':
    inputs = torch.tensor([-2, 2, 1.5])
    targets = torch.tensor([0.0, 1.0, 0.0])
    focal_loss(inputs, targets)

