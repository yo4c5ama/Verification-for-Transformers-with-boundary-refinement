import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np
import json


def read_data(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    return data


def write_file(path, sentences):
    with open(path, 'w', encoding="utf-8") as f:
        for i in sentences:
            f.write(i + '\n')


def normfun(x, mu, sigma):
    function = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return function

def normal_dist(x, mu, sigma, A):
    return A * norm.pdf(x, loc=mu, scale=sigma)

def Histograms_Normal_Distribution(ppl, path):
    mean = np.mean(ppl)
    std = np.std(ppl)
    x = np.arange(0, np.max(ppl), 0.01)
    y = normfun(x, mean, std)

    plt.plot(x, y)

    # plt.hist(ppl, bins=50, rwidth=0.7, density=False)

    # plt.title('perplexity distribution')

    plt.xlabel('Ratio (hsb/de)')
    # plt.xlabel('Perplexity')

    plt.ylabel('Probability Density')
    # plt.ylabel('Number')
    plt.savefig(path)
    plt.show()



def scatter_plot(de_ppl, hsb_ppl, path):

    plt.scatter(x=de_ppl, y=hsb_ppl, alpha=0.01)
    plt.xlabel("de_perplexity")
    plt.ylabel("hsb_perplexity")

    plt.savefig(path)
    plt.show()


def curve_fit_plot(data, path):
    mean = np.mean(data)
    std = np.std(data)
    print("mean:{}".format(mean))
    hist, bins = np.histogram(data, bins=50)

    width = 0.7 * (bins[1] - bins[0])
    # patches, bins, n = plt.hist(data, bins=50, rwidth=0.7, density=False)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width, color='C1')

    popt, pcov = curve_fit(normal_dist, center, hist, maxfev=500000)

    x = np.linspace(center[0], center[-1], 100)
    plt.plot(x, normal_dist(x, *popt), label='fit', color='C0')
    plt.xlabel("Value of epsilon")
    plt.ylabel("Number of words")
    plt.legend(['μ: {:.2f}, σ: {:.2f}'.format(popt[0], popt[1]), 'mean: {:.2f}, std: {:.2f}'.format(mean, std)])
    plt.savefig(path, dpi = 500)
    plt.show()

    pass


def point(a, b):
    plt.figure(figsize=(6, 6))
    plt.scatter(b, a, color='blue', label='data points')  # 横轴是b，纵轴是a


    plt.plot([min(b + a), max(b + a)], [min(b + a), max(b + a)], 'r--', label='y = x')

    plt.xlabel('b (x-axis)')
    plt.ylabel('a (y-axis)')
    plt.title('Scatter Plot with y = x Line')
    plt.legend()

    plt.grid(True)
    plt.axis('equal')  # 保持xy比例一致
    plt.show()

if __name__ == '__main__':
    model = "model_yelp_6"
    p = "1"
    ver_method = ""
    our_ver_method = "_DBR_b"
    # propagate_method = "baf"
    # perturb_words = "1"
    # sentences_number = "100"
    result = './results/{}/p{}/res{}.json'.format(model,p,ver_method)
    our_result = './results/{}/p{}/res{}.json'.format(model,p,our_ver_method)
    pic_path = './results/{}/p{}/epsilon_ratio_distribution({}).jpg'.format(model, p, ver_method)
    def get_epsilon(result_path):
        data = read_data(result_path)
        epsilon = []
        min_eps = 100
        for i in data["examples"]:
            for j in i["bounds"]:
                epsilon.append(j["eps"])
                min_eps = min(min_eps, j["eps"])
        return epsilon, min_eps

    epsilon, min_eps = get_epsilon(result)
    our_epsilon, our_min_eps = get_epsilon(our_result)
    point(our_epsilon, epsilon)
    epsilon_ratio = [x/y for x, y in zip(our_epsilon, epsilon) ]
    time = 0
    # for i in result_data["examples"]:
    #     time += float(i["time"])
    # print("total_time:{}h, avg_time_on_word:{}, min_eps:{}".format(time/3600, time/len(epsilon), min_eps))
    # print(len(epsilon))
    print(len(epsilon_ratio))
    curve_fit_plot(epsilon_ratio, pic_path)
    pass