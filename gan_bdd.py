import torch
import torch.nn.functional as F

import pickle, argparse
import math
import mdd_gen
import discriminator
import generator
import data_utils

import mdd
import numpy as np
np.printoptions(precision=3, suppress=True)


def mask_fill_inf(matrix, mask):
    negmask = 1 - mask
    num = 3.4 * math.pow(10, 38)
    return (matrix * mask) + (-((negmask * num + num) - num))

def train(args):
    ##### variables
    data_file = args.data_file
    prob_file = args.prob_file
    batchnum = args.batchnum
    max_width = args.max_width
    batchnum_summary = args.batchnum_summary
    d_n_hidden = args.d_n_hidden
    g_n_hidden = args.g_n_hidden
    g_n_input = args.g_n_input
    lr = args.lr
    g_lr = args.glr
    use_mask = (args.mask == 1)


    if args.cuda > 0:
        cuda = True
    else:
        cuda = False

    print("read problem")
    prob = pickle.load(open(prob_file, 'rb'))

    # prepare the data gen
    data_gen = data_utils.ScheduleDataGen(data_file, prob.max_stops, prob.num_locs)

    # build mdd
    mdd0 = mdd.MDD_TSP(prob.paired_dist, prob.startp, prob.endp, prob.max_duration, prob.max_stops, max_width)
    mdd0.filter_refine_preparation()
    mdd0.filter_refine()
    mdd0.add_last_node_forever()
    crisp = mdd_gen.MDDTSPMask(mdd0)

    # for seq models
    hidden_dim = g_n_hidden
    z_dim = g_n_input + prob.num_locs
    val_dim = prob.num_locs

    # generators
    seq = generator.SeqV2(hidden_dim, z_dim, val_dim, prob.max_stops + 1)
    if cuda:
        seq = seq.cuda()

    # discriminators
    dmt = discriminator.DRNNV1(d_n_hidden, prob.num_locs)
    if cuda:
        dmt = dmt.cuda()

    d_label_1 = torch.ones(batchnum)
    d_label_0 = torch.zeros(batchnum)
    d_label = torch.cat((d_label_1, d_label_0), 0)
    if cuda:
        d_label = d_label.cuda()

    d_label_1s = torch.ones(batchnum_summary)
    d_label_0s = torch.zeros(batchnum_summary)
    d_label_summary = torch.cat((d_label_1s, d_label_0s), 0)
    if cuda:
        d_label_summary = d_label_summary.cuda()

    d_optim = torch.optim.Adam(dmt.parameters(), lr=lr)
    g_optim = torch.optim.Adam(seq.parameters(), lr=g_lr)

    for epoch in range(1, 500):
        if epoch % 10 == 0:
            print("[", epoch, 'iterations]')
            vars_real, visit = data_gen.next_data(batchnum_summary)

            visit_z0 = torch.Tensor(visit).float()
            visit_z = visit_z0.repeat(prob.max_stops + 1, 1, 1)

            z = torch.randn([prob.max_stops + 1, batchnum_summary, g_n_input])
            z = torch.cat((visit_z, z), 2)

            if cuda:
                z = z.cuda()
            out = seq(z)

            if use_mask:
                out_mask = crisp.generate_mask(out.detach().numpy(), visit_z0)
                out_mask = torch.from_numpy(out_mask)
                out = mask_fill_inf(out, out_mask)
            else:
                if cuda:
                    visit_z = visit_z.cuda()
                out = mask_fill_inf(out, visit_z)

            result = F.softmax(out, dim=2)
            result[torch.isnan(result)] = 0
            vars_synt = result
            if cuda:
                real_routes = data_utils.score_to_routes(vars_real)
                syn_routes = data_utils.score_to_routes(vars_synt.detach().cpu().numpy())
            else:
                real_routes = data_utils.score_to_routes(vars_real)
                syn_routes = data_utils.score_to_routes(vars_synt.detach().numpy())

            print('real=', real_routes[:10])
            print('gen=', syn_routes[:10])

            valid_output = prob.valid_routes(syn_routes, visit)
            print('valid: gen={}'.format(sum(valid_output)*1.0/ (len(visit))))
            norm_reward = prob.norm_reward(syn_routes, real_routes)
            print('reward: norm={}'.format(norm_reward))

        vars_real, visit = data_gen.next_data(batchnum)
        vars_real = torch.Tensor(vars_real).float()
        if cuda:
            vars_real = vars_real.cuda()

        visit_z0 = torch.Tensor(visit).float()
        visit_z = visit_z0.repeat(prob.max_stops + 1, 1, 1)

        z = torch.randn([prob.max_stops + 1, batchnum, g_n_input])
        z = torch.cat((visit_z, z), 2)

        if cuda:
            z = z.cuda()

        out = seq(z)
        if use_mask:
            out_mask = crisp.generate_mask(out.detach().numpy(), visit_z0)
            print("mask: {}".format(out_mask[:,1,:]))
            print("real: {}".format(vars_real[:,1,:]))

            out_mask = torch.from_numpy(out_mask)
            out = torch.mul(out, out_mask) - 1.0 + out_mask
        else:
            out = mask_fill_inf(out, visit_z)

        vars_synt = F.softmax(out, dim=2)
        d_inp = torch.cat((vars_real, vars_synt.detach()), 1)
        d_output = dmt(d_inp)
        d_output = torch.squeeze(d_output)
        d_loss = F.binary_cross_entropy(d_output, d_label)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_output_synt = dmt(vars_synt)
        g_loss = -torch.mean(torch.log(d_output_synt + 1e-10))
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()


if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Generative advarsial network based on MDD.")

    parser.add_argument("--data_file", required=True, help='The data file for training (required).')
    parser.add_argument("--prob_file", required=True, help='The scheduling problem instance (required).')

    parser.add_argument("--batchnum", type=int, help='mini-batch size (default=500)')
    parser.add_argument("--batchnum_summary", type=int, help='mini-batch size for summary (default=batchnum)')
    parser.add_argument("--d_n_hidden", type=int, help='num hidden neurons in the discriminator (default=100).')
    parser.add_argument("--g_n_hidden", type=int, help='num hidden neurons in the RNN of the generator (default=100).')
    parser.add_argument("--g_n_input", type=int, help='num input for the RNN of the generator (default=100).')
    parser.add_argument("--lr", type=float, help='learning rate for the discriminator (default=.5).')
    parser.add_argument("--glr", type=float, help='learning rate for the generator (default=lr).')
    parser.add_argument("--cuda", type=int, help='whether to use cuda (default=0: not use).')
    parser.add_argument("--max_width", type=int, help='max width')
    parser.add_argument("--mask", type=int, help='whether to use mask or not (default=1: use mask).')

    args = parser.parse_args()
    train(args)
