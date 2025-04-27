import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import os
import LDA_SLIC
import CTFN
from functions import train_epoch,test_epoch,output_metric,get_data,normalize

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Pavia', 'Salinas', 'KSC', 'Botswana', 'Houston'],
                    default='Indian', help='dataset to use')
parser.add_argument("--num_run", type=int, default=10)
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--superpixel_scale', type=int, default=80, help='superpixel_scale')# ip 80 compactness=0.06 SA 250 0.005 UP 250 0.005
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='number of seed')
parser.add_argument('--batch_size', type=int, default=256, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU
# -------------------------------------------------------------------------------

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data

input, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true, gt_reshape,y_train_flatten, y_test_flatten= get_data(args.dataset)
##########得到原始图像 训练测试以及所有点坐标 每一类训练测试的个数############

# normalize data by band norm
input_normalize = normalize(input)
height, width, band = input_normalize.shape  # 145*145*200
print("height={0},width={1},band={2}".format(height, width, band))
input_numpy=np.array(input_normalize)
input_normalize = torch.from_numpy(input_numpy.astype(np.float32)).to(device)
# -------------------------------------------------------------------------------

##########得到训练测试以及所有点的光谱############
y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # [695]
Label_train = Data.TensorDataset(y_train,y_train_flatten)

y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # [9671]
Label_test = Data.TensorDataset(y_test,y_test_flatten)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
##########训练集的光谱值及标签##########
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)
##########测试集的光谱值及标签##########

# -------------------------------------------------------------------------------
results = []
for run in range(args.num_run):

    best_OA2 = 0.0
    best_AA_mean2 = 0.0
    best_Kappa2 = 0.0
    best_AA2 = []
    # 获取训练样本的标签图
    train_samples_gt = np.zeros(height*width)#得到21015*1的零数组
    for i in range(len(y_train_flatten)):
        train_samples_gt[y_train_flatten[i]] = gt_reshape[y_train_flatten[i]]
        pass
    #######经过上述操作后train_samples_gt在训练集处获得标签，其余为0########

    ls = LDA_SLIC.LDA_SLIC(input_numpy, np.reshape(train_samples_gt, [height, width]), num_classes - 1)
    Q, S, A, Edge_index, Edge_atter, Seg = ls.simple_superpixel(scale=args.superpixel_scale)
    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)
    Edge_index = torch.from_numpy(Edge_index).to(device)
    Edge_atter = torch.from_numpy(Edge_atter).to(device)
    SP_size=Q.shape[1]

    CNN_nhid = 64  # CNN隐藏层通道数
    net = CTFN.CTFN(height, width, band, num_classes, Q, A, S, Edge_index, Edge_atter, SP_size, CNN_nhid)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # ,weight_decay=0.0001
    criterion = nn.CrossEntropyLoss().cuda()

    torch.cuda.empty_cache()
    for epoch in range(args.epoches):
        net.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(net, input_normalize, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        if (epoch+1) % 10 == 0:
            print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch + 1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1) and epoch >= args.epoches*0.9:

            net.eval()
            tar_v, pre_v = test_epoch(net, input_normalize, label_test_loader, criterion)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            if OA2 >= best_OA2 and AA_mean2 >= best_AA_mean2 and Kappa2 >= best_Kappa2:
                best_OA2 = OA2
                best_AA_mean2 = AA_mean2
                best_Kappa2 = Kappa2
                best_AA2 = AA2

    torch.cuda.empty_cache()

    print("\nbest_OA:{:.2f}, best_AA:{:.2f}, best_Kappa:{:.2f}".format(best_OA2*100, best_AA_mean2*100, best_Kappa2*100))
    f = open('./result/' + str(args.dataset) + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + '\nOA={:.2f}'.format(best_OA2*100) \
                  + '\nAA={:.2f}'.format(best_AA_mean2*100) \
                  + '\nKappa={:.2f}'.format(best_Kappa2*100) \
                  + '\nbest_AA2=' + str(np.around(best_AA2 * 100, 2))

    f.write(str_results)
    f.close()











