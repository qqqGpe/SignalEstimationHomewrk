import cv2
import numpy as np
import random
import math
# from sklearn.mixture import GaussianMixture


# class GaussianModel():
#     def __init__(self):
#         self.miu = random.uniform(0,1)
#         self.sigma = random.uniform(0,1)

class GMM_model():
    def __init__(self, component_num, max_iter, eps):
        self._component_num = component_num
        self._max_iter = max_iter
        self._eps = eps
        self.gaussModelParam = np.zeros((self._component_num, 3))
        self.gamma_hat = np.zeros((len(x), self._component_num))

        # self.alpha = np.random.random(self._component_num)
        # self.alpha = self.alpha / sum(self.alpha)
        for i in range(component_num):
            # self.gaussModelParam.append([random.uniform(0,1), random.uniform(1,2)]) # init miu and sigma
            # self.gaussModelParam.append([random.uniform(0,1), random.uniform(1,2), 1./3]) # init miu, sigma and alpha
            self.gaussModelParam[i, :] = np.array([random.uniform(0,1), random.uniform(1,2), 1./3])

    def difference(self, previousParam)->float:
        # for i in range(len(self.gaussModelParam)):
        #     diff += ((previousParam[i][0] - self.gaussModelParam[i][0])**2 + (previousParam[i][1] - self.gaussModelParam[i][1])**2)**0.5
        # diff = diff / len(self.gaussModelParam)
        diff = np.linalg.norm(previousParam - self.gaussModelParam)
        return diff

    def predict(self, observation):
        x = observation
        diff = 100.
        iter_num = 0
        while diff > self._eps and iter_num <= self._max_iter:
            iter_num += 1
            paramPre = np.copy(self.gaussModelParam)
            # E step
            for j in range(len(x)):
                p_sum = 0.
                for k in range(self._component_num):
                    miu_k = self.gaussModelParam[k][0]
                    sigma_k = self.gaussModelParam[k][1]
                    alpha_k = self.gaussModelParam[k][2]
                    self.gamma_hat[j][k] = alpha_k * (1.0 / (math.sqrt(2.0*math.pi)*sigma_k)) \
                                              * math.exp(-(x[j] - miu_k)**2.0 / (2.0*sigma_k**2))
                    p_sum += self.gamma_hat[j][k]
                self.gamma_hat[j, :] = self.gamma_hat[j, :] / p_sum
            # M step
            for k in range(self._component_num):
                # calculate miu_k_hat
                miu_sum_up = 0.
                miu_sum_down = 0.
                sigma_sum_up = 0.
                alpha_sum_down = len(x)
                for j in range(len(x)):
                    miu_sum_up += self.gamma_hat[j][k] * x[j]
                    miu_sum_down += self.gamma_hat[j][k]
                    sigma_sum_up += self.gamma_hat[j][k] * (x[j] - self.gaussModelParam[k][0])**2
                sigma_sum_down = miu_sum_down
                alpha_sum_up = miu_sum_down
                self.gaussModelParam[k][0] = miu_sum_up / miu_sum_down
                self.gaussModelParam[k][1] = (sigma_sum_up / sigma_sum_down)**0.5
                self.gaussModelParam[k][2] = alpha_sum_up / alpha_sum_down
            diff = self.difference(paramPre)
            print('Iteration Num = %d' % iter_num)
        print('Prediction finished!')


def preprocess(x):
    return (x - x.mean(keepdims=True)) / x.std(keepdims=True)

# EM hyper parameters
epsilon = 1e-4 # stopping criterion
R = 10 # number of re-runs
N = 2 # number of components
max_iter = 30 # stopping criterion max iterations

# read in the example image
img = cv2.imread('.\\1.jpg')
orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = preprocess(np.copy(orig))
x = img.reshape(img.shape[0]*img.shape[1],-1)

gmm = GMM_model(component_num=N, max_iter=max_iter, eps=epsilon)
gmm.predict(x)

clustering = np.copy(gmm.gamma_hat)
classification = np.zeros((img.shape[0], img.shape[1]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if(clustering[i*j][0] > 0.5):
            classification[i][j] = 0
        else:
            classification[i][j] = 255

from matplotlib import pyplot as plt
# show results
plt.imshow(classification)
plt.show()

pass

