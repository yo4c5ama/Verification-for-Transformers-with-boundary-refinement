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


# def point(a, b, pic_path):
#     plt.figure(figsize=(6, 6))
#     # plt.scatter(b, a, color='blue', label='data points')  # 横轴是b，纵轴是a
#     plt.scatter(b, a, color='skyblue', alpha=0.5, s=10, label='data points')
#
#     plt.plot([min(b + a), max(b + a)], [min(b + a), max(b + a)], 'r--', label='y = x')
#
#     plt.xlabel('b (x-axis)')
#     plt.ylabel('a (y-axis)')
#     plt.title('Scatter Plot with y = x Line')
#     plt.legend()
#
#     plt.grid(True)
#     plt.axis('equal')  # 保持xy比例一致
#     plt.savefig(pic_path, dpi=300)
#     plt.show()

def point(a, b, pic_path):
    plt.figure(figsize=(6, 6), dpi=300)

    # 散点
    for n,m,cl in zip(a,b,['1','2','∞']):
        plt.scatter(m, n, alpha=0.5, s=10, label='p={}'.format(cl))

    # y=x 线
    min_val = min(min(a[0]), min(b[0]))
    max_val = max(max(a[0]), max(b[0]))
    plt.plot([0, 1], [0, 1], linestyle='--',color="red", label='y = x', linewidth=1.5)

    # 坐标轴标签和标题
    plt.xlabel('epsilon value of baseline', fontsize=12)
    plt.ylabel('epsilon value of ours', fontsize=12)
    plt.title('Scatter Plot with y = x Line and N=6 in sst', fontsize=14)

    # # 设置坐标轴从 0 开始
    # plt.xlim(left=1)
    # plt.ylim(bottom=1)

    # 坐标轴线条美化
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 网格线美化
    plt.grid(color='gray', linestyle='--', linewidth=1)

    # 图例
    plt.legend()

    # 坐标轴比例一致
    plt.axis('equal')

    # 自动布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(pic_path, dpi=300, bbox_inches='tight')  # 避免边缘裁切
    plt.show()

def data_processing(our_epsilon,epsilon):
    our_DBR = []
    baseline = []
    our_time = []
    base_time = []
    for our, base in zip(our_epsilon,epsilon):
        a = []
        b = []
        a_t = []
        b_t = []
        DBR_e = []
        e = []
        for i in our:
            for j in base:
                if i[0] == j[0]:
                    if len(i)==len(j):
                        a_t.append(i[1])
                        b_t.append(j[1])
                    a.append(i[2:])
                    b.append(j[2:len(i)])
                    break
                else:
                    continue
        DBR_e = [item for sublist in a for item in sublist]
        e = [item for sublist in b for item in sublist]
        our_DBR.append(DBR_e / max(np.max(DBR_e), np.max(e)))
        baseline.append(e / max(np.max(DBR_e), np.max(e)))
        our_time.append(a_t)
        base_time.append(b_t)
    return our_DBR, baseline, our_time, base_time

if __name__ == '__main__':
    model = "model_sst_6"
    p = ["1","2","100"]
    ver_method = ""
    our_ver_method = "_DBR"
    # propagate_method = "baf"
    # per_turbwords = "1"
    # sentences_number = "100"
    result = ['./results/{}/p{}/res{}.json'.format(model,i,ver_method) for i in p]
    our_result = ['./results/{}/p{}/res{}_b.json'.format(model,i,our_ver_method) for i in p]
    result_2 = ['./results/{}/p{}/res_2{}.json'.format(model,i,ver_method) for i in p]
    our_result_2 = ['./results/{}/p{}/res_2{}.json'.format(model, i, our_ver_method) for i in p]
    pic_path = './results/{}/epsilon_ratio_distribution({}).jpg'.format(model, ver_method)
    def get_epsilon(result_path):
        E_results = []
        min_E = []
        for path in result_path:
            data = read_data(path)
            epsilon = []
            min_eps = 100
            for n,i in enumerate(data["examples"]):
                epsilon.append([])
                epsilon[n].append(i["tokens"])
                epsilon[n].append(i["time"])
                for j in i["bounds"]:
                    epsilon[n].append(j["eps"])
                    min_eps = min(min_eps, j["eps"])
            E_results.append(epsilon)
            min_E.append(min_eps)
        return E_results, min_E

    epsilon, min_eps = get_epsilon(result)
    our_epsilon, our_min_eps = get_epsilon(our_result)
    DBR_e, e, DBR_e_time, e_min_time = data_processing(our_epsilon,epsilon)
    epsilon_2, min_eps_2 = get_epsilon(result_2)
    our_epsilon_2, our_min_eps_2 = get_epsilon(our_result_2)
    DBR_e_2, e_2, DBR_e_time_2, e_min_time_2 = data_processing(our_epsilon_2,epsilon_2)
    DBR_e = [list(row_a) + list(row_b) for row_a, row_b in zip(DBR_e, DBR_e_2)]
    e = [list(row_a) + list(row_b) for row_a, row_b in zip(e, e_2)]
    point(DBR_e, e, pic_path)
    # epsilon_ratio = [x/y for x, y in zip(DBR_e, e) ]
    time = 0
    # print(len(DBR_e), np.min(DBR_e), np.mean(DBR_e),
    #       np.mean(DBR_e_time))
    # print(len(epsilon_ratio),np.min(epsilon_ratio),np.mean(epsilon_ratio),np.mean([x/y for x, y in zip(DBR_e_time, e_min_time)]))
    # curve_fit_plot(epsilon_ratio, pic_path)
    pass