# smo
import numpy as np
from numpy import *


def loadDataSet(filename):
    X = []
    Y = []
    f = open(filename)
    for line in f.readlines():
        linearr = line.strip().split('\t')
        X.append([float(linearr[0]), float(linearr[1])])
        Y.append(float(linearr[2]))
    return mat(X), mat(Y).T


class SMO:
    def __init__(self, X, Y, C, e, gaussDelta=1.3):
        # gaussDelta：高斯核函数的分母
        self.X = X
        self.Y = Y
        self.C = C
        # 容错率
        self.tol = e
        # 样本个数
        self.N = shape(X)[0]
        self.alphas = mat(zeros((self.N, 1)))
        self.b = 0
        # 整个数据集的非边界的E缓存
        self.ECache = mat(zeros((self.N, 2)))
        # k_ij的存储地
        self.K = mat(zeros((self.N, self.N)))
        # i,j都是X里面的一个样本
        for i in range(self.N):
            self.K[:, i] = self.kernelGauss(self.X, self.X[i, :],gaussDelta)

    def kernelGauss(self, X, Y, delta):
        m, n = shape(X)
        k = mat(zeros((m, 1)))
        for j in range(m):
            diff = X[j, :] - Y
            k[j] = diff * diff.T
        return exp(-k / (2 * math.pow(delta, 2)))

    def calcKernel(self, i, j):
        return self.K[i, j]

    def calcEi(self, i):
        # 这里的i是选取的自由变量，i表示第i维度
        # K(x_j,x_i)
        ki = self.K[:, i]
        aiyi = multiply(self.alphas, self.Y) # multuply对应位置相乘即可
        fxi = float(aiyi.T * ki) + self.b
        Ei = fxi - float(self.Y[i])
        return Ei

    def selectJ(self, i, Ei):
        j = -1
        maxDeltaE = 0
        Ej = 0
        self.ECache[i] = [1, Ei]
        # 要求选择的Ej首先是非边界值，否则返回随机值
        # .A 矩阵变成数组,nonzero 返回非0数值的下标键值对
        # 比如(array([1, 3]), array([0, 0]))意思是(1,0)和(3,0)位置是非0的
        # 由于只有一列，所以只需要知道第一个array即可，后面的都为0
        validEcacheList = nonzero(self.ECache[:, 0].A)[0]
        if len(validEcacheList) > 1:    # 非
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEi(k)
                # |Ei-Ej|最大
                deltaE = abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxDeltaE = deltaE
                    Ej = Ek
                    j = k
            return j, Ej
        else:
            j = self.selectJRand(i, self.N)
            Ej = self.calcEi(j)
            return j, Ej

    def selectJRand(self, i, n):
        j = i
        while j == i:
            j = int(random.uniform(0, n))
        return j

    def updateEi(self, i):
        Ei = self.calcEi(i)
        self.ECache[i] = [1, Ei]

    def updateAlphaB(self, i, e=0.00001):
        Ei = self.calcEi(i)
        # 先选择在0<ai<C上的点，看是否满足kkt
        # yi*g(xi)=y(Ei+yi)=>yi*Ei=0            满足kkt
        # 加上容错率：yi*Ei=0+tol or yi*Ei=0-tol  满足kkt
        # 选择那些在容错率之外的样本作为第一个变量
        # if条件表达为违反kkt条件的非边界值
        if ((self.Y[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                ((self.Y[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            aiOld = self.alphas[i].copy()
            ajOld = self.alphas[j].copy()   # -1就会是最后一个值，不会报错
            # 这里的处理带了一些巧合，如果第一次训练集yi和yj相同，测试数据集变得很差
            # 也不能说是错误，因为至少一个ai违反，也是对的
            # 这里的eta没有严格按照论文来写
            if self.Y[i] != self.Y[j]:
                L = max(0, ajOld - aiOld)
                H = min(self.C, self.C + ajOld - aiOld)
            else:
                L = max(0, aiOld + ajOld - self.C)
                H = min(self.C, aiOld + ajOld)
            if L == H:
                print("L == H")
                return 0
            eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
            # eta 类似于二阶导数值，只有当它 大于 0 才能取最小值
            if eta <= 0:
                print("eta <= 0")
                return 0
            # 计算 alpha_j 并截取在[H,L]之内
            self.alphas[j] = ajOld + self.Y[j] * (Ei - Ej) / eta
            self.alphas[j] = self.truncateAlpha(self.alphas[j], H, L)
            self.updateEi(j)
            # 无法满足足够的下降
            if abs(self.alphas[j] - ajOld) < e:
                print("j not moving enough")
                return 0
            self.alphas[i] = aiOld + self.Y[i] * self.Y[j] * \
                            (ajOld - self.alphas[j])
            self.updateEi(i)

            # 更新 b 的值
            b1 = -Ei - self.Y[i] * self.K[i, i] * (self.alphas[i] - aiOld) \
                 - self.Y[j] * self.K[j, i] * (self.alphas[j] - ajOld) + self.b

            b2 = -Ej - self.Y[j] * self.K[i, j] * (self.alphas[i] - aiOld) \
                 - self.Y[j] * self.K[j, j] * (self.alphas[j] - ajOld) + self.b

            if 0 < self.alphas[j] < self.C:
                self.b = b1
            elif 0 < self.alphas[i] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def truncateAlpha(self, aj, H, L):
        if aj > H:
            return H
        elif aj < L:
            return L
        else:
            return aj

    def train(self, maxIter):
        iter = 0    # 迭代次数
        entireSet = True
        alphaPairsChanged = 0
        # 在整个数据集和非边界值上面来回切换选取变量
        # 先单次遍历整个数据集，去试试有没有违反kkt的非边界值可以改变
        # 当发现有非边界值改变时就单独遍历非边界集
        # 如果还有改变就继续第三步，没有则进行第二步，遍历整个数据集
        while (iter < maxIter) and ((alphaPairsChanged > 0) or entireSet):
            alphaPairsChanged = 0
            # 遍历所有值
            if entireSet:
                for i in range(self.N):
                    alphaPairsChanged += self.updateAlphaB(i)
                    print("fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
                    iter += 1
            # 遍历非边界值
            else:
                # 在 alphas 中取出大于0且小于c的索引值
                # 这里的*相当于and
                nonBoundIs = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.updateAlphaB(i)
                    print("non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif alphaPairsChanged == 0:
                entireSet = True
            print("iteration number: %d" % iter)
        return self.b, self.alphas


if __name__ == '__main__':
    filename = 'testSetRBF.txt'
    X, Y = loadDataSet(filename)
    smo = SMO(X, Y, 200, 0.0001)
    b, alphas = smo.train(10000)
    # 支持向量的索引
    svIndices = nonzero(smo.alphas.A > 0)[0]
    # 支持向量特征
    xSv = X[svIndices]
    # 支持向量分类
    ySv = Y[svIndices]
    print("there are %d Support Vectors" % shape(xSv)[0])
    m, n = shape(X)
    errorCount = 0
    delta = 1.3
    for i in range(m):
        # 映射到核函数值
        kernelEval = smo.kernelGauss(xSv, X[i, :], delta)
        predict = kernelEval.T * multiply(ySv, smo.alphas[svIndices]) + b
        if sign(predict) != sign(Y[i]):
            errorCount += 1
    # 训练样本误差
    print("the training error rate is : %f" % (float(errorCount / m)))

    errorCount = 0
    X, Y = loadDataSet("testSetRBF2.txt")
    m, n = shape(X)
    for i in range(m):
        kernelEval = smo.kernelGauss(xSv, X[i, :], delta)
        predict = kernelEval.T * multiply(ySv, smo.alphas[svIndices]) + b
        if sign(predict) != sign(Y[i]):
            errorCount += 1

    # 测试样本误差
    print("the test error rate is : %f" % (float(errorCount / m)))
