import matplotlib.pyplot as plt
import numpy as np

def acc_with_global_rounds(acc,args):
    # 创建x轴（例如，使用训练的epoch数）
    epochs = range(1, len(acc) + 1)

    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, linestyle='-', color='b', label='Accuracy')

    # 添加标题和标签
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # plt.grid(True)
    plt.xticks(epochs)

    # 显示图例
    plt.legend()

    # 显示图形
    plt.tight_layout()
    # plt.show()
    plt.savefig('save/'+args.dataset+'_defense_'+str(args.epsilon)+"_"+str(args.cur_experiment_no)+'.jpg')



def mean_params_and_metric_and_random(a,b,random):
    # 创建画布和第一个子图
    fig, ax1 = plt.subplots()

    # 绘制第一个数据集
    ax1.plot(a, 'g-', label='Y1 mean_params')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y1 mean_params', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # 创建第二个子图并绑定到同一X轴
    ax2 = ax1.twinx()
    ax2.plot(b, 'b-', label='Y2 all_l2_norm')
    ax2.set_ylabel('Y2 all_l2_norm', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # 添加第三条线到右边的y轴
    ax2.plot(random, 'r-', label='Random')
    ax2.tick_params(axis='y', labelcolor='r')

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    # 保存图像
    # plt.savefig("save/image.png")

    # 显示图形
    plt.show()

def attack_performance(a,b,c=None):
    plt.xlabel('global round')
    plt.ylabel('metric')
    plt.plot(b,label = 'no active attack')
    d = [0.018918092979216854, 0.022487160781558185, 0.022504256038821044, 0.022504435461128014, 0.022504388532841715, 0.022504326966818724, 0.022504445260942856, 0.022504381880986607, 0.0225043842225359, 0.022504385115465552, 0.022504326826123584, 0.02250438604658944, 0.02250433459416365, 0.022504384880577995, 0.022504439625329525, 0.02250432812499453, 0.02250438492352441, 0.022504445034787407, 0.02250438396381519, 0.022504328686338174, 0.022504436746844703, 0.022504387041771715, 0.02250432500959659, 0.022504383334741737, 0.022504385403600145, 0.022504382449346168, 0.022504385804113007, 0.022504327908697743, 0.022504391569570688, 0.022504336205145296, 0.02250438632713228, 0.02250433801932959, 0.022504442801625873, 0.022504327931972493, 0.022504385843520137, 0.022504382673862018, 0.02250433082804544, 0.022504383685172474, 0.02250438406190704, 0.02250432772087003, 0.022504383599411315, 0.02250438389987329, 0.02250433167550267, 0.02250443449699266, 0.022504378069224593, 0.0225042701328986, 0.022504329949044374, 0.022504383842199747, 0.022504329016712284, 0.022504442667869082, 0.022504332584053755, 0.022504318821061832, 0.022504386914464398, 0.022504324447180722, 0.022504333760811786, 0.02250433153086716, 0.022504384315832012, 0.022504385265343395, 0.022504443706566624, 0.022504382806032078, 0.022504389385107345, 0.02250438236409061, 0.022504386472906406, 0.02250438759166915, 0.022504383177713496, 0.022504435862650593, 0.022504386432436536, 0.022504328300991604, 0.02250438791717973, 0.02250438241532502, 0.02250432401276414, 0.022504377507186264, 0.0225043273368061, 0.0225042747318141, 0.022504383760744272, 0.02250433645274132, 0.022504380880609246, 0.02250438377365095, 0.02250438418099608, 0.022504329939738006, 0.02250438827064707, 0.022504334302483532, 0.022504383349287903, 0.022504388638083225, 0.022504387389482978, 0.022504331782064155, 0.022504334913404124, 0.022504278393884466, 0.022504382352453285, 0.02250443880223103, 0.0225043325353459, 0.022504331310598927, 0.022504275041049585, 0.02250438800117198, 0.02250444007227068, 0.02250438804751904, 0.02250432942405591, 0.022504329990400005, 0.02250433522260216, 0.022504329385730916]
    # d=[1,2,3]
    if c != None:
        plt.plot(c,label='random')
    plt.plot(d,label='active attack')
    plt.legend(loc = 'upper right')
    plt.show()

def two_metric(a,b):
    length = len(a)
    x = [i for i in range(length)]
    plt.xlabel('Global Round')
    plt.ylabel('Metric')
    plt.plot(x,a,label = 'Cos-sim')
    plt.plot(x,b,label = 'JS-div')
    plt.legend()
    plt.show()

def two_diff_y_axis(a,b,y1,y2,title):
    length = len(a)
    x = [i for i in range(length)]
    fig, ax1 = plt.subplots()

    # 绘制左边的折线
    ln1 = ax1.plot(x, a, color='#df7a5e',label = y1,linewidth=2.5)
    ax1.set_xlabel('Global Round')
    ax1.set_ylabel(y1, color='#df7a5e')
    ax1.tick_params(axis='y', labelcolor='#df7a5e')

    # 创建第二个纵轴对象
    ax2 = ax1.twinx()
    #5B9BD5
    # 绘制右边的折线
    ln2 = ax2.plot(x, b, color='#5B9BD5',label = y2,linewidth=2.5)
    ax2.set_ylabel(y2, color='#5B9BD5')
    ax2.tick_params(axis='y', labelcolor='#5B9BD5')

    
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns,labs,loc = 'center right')

    # fig.legend(lns,labs,loc = 'upper left',ncol = 2)
    # 设置图表标题
    ax1.set_title(title)
    # 显示图表
    plt.show()

def single_y_axi(a,b=None,c=None):
    length = len(a)
    x = [i for i in range(length)]

    plt.plot(x,a)


def two_metric_with_C(cos_sim,js_div,dataset):
    # length = len(a)
    plt.title(dataset,loc='center')
    x = [0.01,0.05,0.1,0.5,1,5,10,50,100]
    plt.xlabel('C')
    plt.ylabel('Metric')
    plt.plot(x,cos_sim,label = 'Cos-sim')
    plt.plot(x,js_div,label = 'JS-div')
    plt.legend()
    # plt.show()
    plt.savefig('save/'+dataset+"_metric_with_C.jpg",dpi=300)

def two_metric_with_C(cos_sim, js_div, dataset):
    x = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

    fig, ax1 = plt.subplots()

    ax1.set_title(dataset, loc='center')
    ax1.set_xlabel('C')
    ax1.set_ylabel('Cos-sim')
    ax1.plot(x, cos_sim, label='Cos-sim', color='b', marker='o')

    ax2 = ax1.twinx()
    ax2.set_ylabel('JS-div')
    ax2.plot(x, js_div, label='JS-div', color='r', marker='s')

    fig.tight_layout()
    fig.legend(loc='upper right')

    plt.savefig('save/' + dataset + "_metric_with_C.jpg", dpi=300)

def acc_with_two_metric_mean_var(mean_acc,var_acc,mean_cos_sim,var_cos_sim,mean_js_div,var_js_div,dataset,font_size=10):
    if mean_acc == []:
        input_file = 'why_active_WikiCS_300_2_5_0.5.txt'
        with open(input_file, 'r') as f:
            current_list = None
            for line in f:
                line = line.strip()  # 去除每行末尾的换行符
                if line.startswith('mean_acc:'):
                    current_list = mean_acc
                elif line.startswith('var_acc:'):
                    current_list = var_acc
                elif line.startswith('mean_cos_sim:'):
                    current_list = mean_cos_sim
                elif line.startswith('var_cos_sim:'):
                    current_list = var_cos_sim
                elif line.startswith('mean_js_div:'):
                    current_list = mean_js_div
                elif line.startswith('var_js_div:'):
                    current_list = var_js_div
                elif line != '':
                    current_list.append(float(line))

    epochs = [i for i in range(len(mean_acc))]
    epochs = np.array(epochs)
    mean_acc = np.array(mean_acc)
    var_acc = np.array(var_acc)
    mean_cos_sim = np.array(mean_cos_sim)
    var_cos_sim = np.array(var_cos_sim)
    mean_js_div = np.array(mean_js_div)
    var_js_div = np.array(var_js_div)
    # 创建一个包含两个子图的图形，左右排列

    # 设置全局字体大小
    plt.rcParams.update({'font.size': font_size})  # 设置所有字体的大小为12

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    ax1.set_title('Model accuracy and cos-sim')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ln1 = ax1.plot(epochs, mean_acc,linewidth =0.8  ,color='#2596be', label='Accuracy')
    # ax1.fill_between(epochs, mean_acc - var_acc, mean_acc + var_acc, alpha=0.2, color='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cos-sim')
    ln2 = ax2.plot(epochs, mean_cos_sim,linewidth =0.8, color='#ed6825', label='Cos-sim')
    # ax2.fill_between(epochs, mean_cos_sim - var_cos_sim, mean_cos_sim + var_cos_sim, alpha=0.2, color='r')
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns,labs,loc = 'center right',prop = {'size':8})

    ax3.set_title('Model accuracy and JS-div')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ln3 = ax3.plot(epochs, mean_acc,linewidth =0.8, color='#2596be', label='Accuracy')
    # ax3.fill_between(epochs, mean_acc - var_acc, mean_acc + var_acc, alpha=0.2, color='b')
    ax4 = ax3.twinx()
    ax4.set_ylabel('JS-div')
    ln4 = ax4.plot(epochs, mean_js_div,linewidth =0.8, color='#35b777', label='JS-div')
    # ax4.fill_between(epochs, mean_js_div - var_js_div, mean_js_div + var_js_div, alpha=0.2, color='g')
    lns2 = ln3+ln4
    labs2 = [l.get_label() for l in lns2]
    ax3.legend(lns2,labs2,loc = 'center right',prop = {'size':8})

    plt.tight_layout()
    # plt.savefig('save/'+dataset+'_mean_var_acc.pdf',format='pdf')
    plt.show()



if __name__ == '__main__':
    dataset = 'WikiCS'
    # cos_sim = [1.000,1.000,1.000,1.000,1.000,0.998,0.979,0.885,0.885]
    # js_div = [0.000,0.000,0.000,0.000,0.000,0.001,0.012,0.045,0.045]
    # two_metric_with_C(cos_sim,js_div,dataset)
    acc_with_two_metric_mean_var([],[],[],[],[],[],dataset,14)