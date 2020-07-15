import torch
import torch.nn.functional as F

import pickle, argparse
import math
import CRISP_TSP_MASK
from generative_adversarial_network import discriminator, generator

import data_utils

import mdd
import numpy as np
np.printoptions(precision=3, suppress=True)



def mask_fill_inf(matrix, mask):
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (matrix * mask) + (-((negmask * num + num) - num))


def test(data_gen, testing_set_size, g_n_input, cuda, crisp, use_mask, gnt):
    real_routes=[]
    predicted_routes=[]
    num_valid_routes=0
    norm_reward=0.0
    for i in range(testing_set_size):
        _, visit, real_paths = data_gen.next_data(1)
        visit_z0 = torch.Tensor(visit).float()
        visit_z = visit_z0.repeat(prob.max_stops + 1, 1, 1)

        z = torch.randn([prob.max_stops + 1, 1, g_n_input])
        z = torch.cat((visit_z, z), 2)

        if cuda:
            z = z.cuda()
        out = gnt(z)
        vars_predict = F.softmax(out, dim=2)
        if use_mask:
            predicted_route = crisp.generate_route_with_inference(vars_predict.detach().numpy(), visit_z0)
        else:
            if cuda:
                visit_z = visit_z.cuda()
            # predicted_route = mask_fill_inf(out, visit_z)
        real_route=real_paths.transpose()
        num_valid_routes += prob.valid_routes([predicted_route], visit)

        norm_reward += prob.norm_reward([predicted_route], real_route)
        real_routes.append(list(real_route.squeeze()))
        predicted_routes.append(predicted_route)

    print('real={}'.format(real_routes[:10]))
    print('gen={}'.format(predicted_routes[:10]))
    print('valid routes={}'.format(num_valid_routes[0] * 1.0 /testing_set_size))
    print('norm-reward={}'.format(norm_reward * 1.0 /testing_set_size))


def build_crisp(prob, max_width):
    # build mdd
    mdd0 = mdd.MDD_TSP(prob.paired_dist, prob.startp, prob.endp, prob.max_duration, prob.max_stops, max_width)
    mdd0.filter_refine_preparation()
    mdd0.relax_mdd()
    mdd0.add_last_node_forever()
    crisp = CRISP_TSP_MASK.CRISP_TSP_MASK(mdd0)
    return crisp


def train(data_file, prob, train_batch_size, max_width, testing_set_size, d_n_hidden, g_n_hidden, g_n_input,
          lr, g_lr, use_crisp, cuda):
    # prepare the data gen
    data_gen = data_utils.ScheduleDataGen(data_file, prob.max_stops, prob.num_locs)
    crisp = build_crisp(prob, max_width)

    # generators
    gnt = generator(hidden_dim=g_n_hidden, z_dim=g_n_input + prob.num_locs, val_dim=prob.num_locs,
                    n_vars=prob.max_stops + 1)
    if cuda:
        gnt = gnt.cuda()
    # discriminators
    dmt = discriminator(hidden_dim=d_n_hidden, val_dim=prob.num_locs)
    if cuda:
        dmt = dmt.cuda()

    d_label_1 = torch.ones(train_batch_size)
    d_label_0 = torch.zeros(train_batch_size)
    d_label = torch.cat((d_label_1, d_label_0), 0)
    if cuda:
        d_label = d_label.cuda()

    d_optim = torch.optim.Adam(dmt.parameters(), lr=lr)
    g_optim = torch.optim.Adam(gnt.parameters(), lr=g_lr)

    for epoch in range(1, 50):
        print("[ {} iterations]".format(epoch))
        for it in range(100):
            vars_real, visit, real_paths = data_gen.next_data(train_batch_size)
            vars_real = torch.Tensor(vars_real).float()
            visit_z0 = torch.Tensor(visit).float()
            if cuda:
                vars_real = vars_real.cuda()

            visit_z = visit_z0.repeat(prob.max_stops + 1, 1, 1)
            # the random variable z
            z = torch.randn([prob.max_stops + 1, train_batch_size, g_n_input])
            z = torch.cat((visit_z, z), 2)
            if cuda:
                z = z.cuda()

            out = gnt(z)
            vars_predict = F.softmax(out, dim=2)
            if use_crisp:
                out_mask=crisp.generate_mask_with_ground_truth(real_paths, visit_z0)

                out_mask = torch.from_numpy(out_mask)
                # mask
                # print("out mask:",out_mask)
                vars_predict = torch.mul(vars_predict, out_mask)
                # print("after masking:",vars_predict)
                # renormalize
                norm_vars_predict=vars_predict/torch.sum(vars_predict, dim=2, keepdim=True)
                # print("after noralization:",norm_vars_predict)
                norm_vars_predict[torch.isnan(norm_vars_predict)] = 0.0
            else:
                norm_vars_predict = vars_predict
            # feed the correct route and the predicted route into the discriminator.
            d_inp = torch.cat((vars_real, norm_vars_predict.detach()), 1)
            d_output = dmt(d_inp)
            d_output = torch.squeeze(d_output)
            d_loss = F.binary_cross_entropy(d_output, d_label)

            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            d_output_predict = dmt(vars_predict)
            g_loss = -torch.mean(torch.log(d_output_predict + 1e-10))
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
        # used for testing
        test(data_gen, testing_set_size, g_n_input, cuda, crisp, use_crisp, gnt)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative adversarial network with MDD.")

    parser.add_argument("--data_file", required=True, help='The data file for training (required).')
    parser.add_argument("--prob_file", required=True, help='The scheduling problem instance (required).')

    parser.add_argument("--batchnum", type=int, help='mini-batch size (default=500)')
    parser.add_argument("--batchnum_summary", type=int, help='mini-batch size for summary (default=batchnum)')
    parser.add_argument("--d_n_hidden", type=int, help='num hidden neurons in the discriminator (default=100).')
    parser.add_argument("--g_n_hidden", type=int, help='num hidden neurons in the RNN of the generator (default=100).')
    parser.add_argument("--g_n_input", type=int, help='num input for the RNN of the generator (default=100).')
    parser.add_argument("--lr", type=float, help='learning rate for the discriminator (default=.5).')
    parser.add_argument("--glr", type=float, help='learning rate for the generator (default=lr).')
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use cuda (default=0: not use).")
    parser.add_argument("--max_width", type=int, help='max width')
    parser.add_argument("--use_crisp", type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use crisp or not (default=1: use crisp).")

    args = parser.parse_args()
    print("read problem")
    prob = pickle.load(open(args.prob_file, 'rb'))
    train(args.data_file, prob, args.batchnum, args.max_width, args.batchnum_summary, args.d_n_hidden, args.g_n_hidden,
          args.g_n_input, args.lr, args.glr, args.use_crisp, args.cuda)
