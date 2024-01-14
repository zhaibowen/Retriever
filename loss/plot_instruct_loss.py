import matplotlib.pyplot as plt

def loss_parser(file):
    with open(file, 'r') as f:
        x = f.readlines()
        x = list(filter(lambda x: x[:4] == "step", x))
        y = list(map(lambda x: x.split("loss ")[1].split(',')[0], x))
        y = list(map(float, y))
        z = list(map(lambda x: x.split("step ")[1].split(',')[0], x))
        z = list(map(int, z))
    return z, y

def main():
    file = "/home/work/disk/vision/retriever/loss/instruct_retriever_tv2_110M_8B_loss1.28.txt"
    step, loss = loss_parser(file)

    # file2 = "/home/work/disk/vision/retriever/loss/retriever_tv2_396M_96B_loss2.40.txt"
    # step2, loss2 = loss_parser(file2)

    # file3 = "/home/work/disk/vision/retriever/loss/retriever_small_step8w_batch256_loss3.48.txt"
    # step3, loss3 = loss_parser(file3)

    # file4 = "/home/work/disk/vision/retriever/loss/retriever_small_step8w_loss3.51.txt"
    # step4, loss4 = loss_parser(file4)

    # file5 = "/home/work/disk/vision/retriever/loss/retriever_small_step8w_mlp_soft_silu__intersize_rotary_loss3.36.txt"
    # step5, loss5 = loss_parser(file5)

    # file6 = "/home/work/disk/vision/retriever/loss/retriever_35M_42B_loss3.29.txt"
    # step6, loss6 = loss_parser(file6)

    # file7 = "/home/work/disk/vision/retriever/loss/retriever_110M_100B_loss2.94.txt"
    # step7, loss7 = loss_parser(file7)

    plt.plot(step, loss)
    # plt.plot(step2, loss2)
    # plt.plot(step3, loss3)
    # plt.plot(step4, loss4)
    # plt.plot(step5, loss5)
    # plt.plot(step6, loss6)
    # plt.plot(step7, loss7)
    # plt.xlim(right=160000)
    # plt.ylim(bottom=2.2, top=4)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()