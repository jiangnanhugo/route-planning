import torch
import torch.nn.functional as F
import numpy as np
import pickle
import argparse

from generative_adversarial_network import discriminator, generator
import data_utils

from mdd.crisp import CRISP
np.printoptions(precision=3, suppress=True)


def test(data_gen, testing_set_size, g_n_input, crisp, use_crisp, use_post_process_type1,use_post_process_type2, gnt):
    real_routes = []
    predicted_routes = []
    num_valid_routes = []
    norm_reward = []
    for i in range(testing_set_size):
        _, visit, real_paths = data_gen.next_data(1)
        visit_z0 = torch.Tensor(visit).float()
        visit_z = visit_z0.repeat(prob.max_stops + 1, 1, 1)
        z = torch.randn([prob.max_stops + 1, 1, g_n_input])
        z = torch.cat((visit_z, z), 2)

        out = gnt(z)
        vars_predict = F.softmax(out, dim=2)
        if use_crisp:
            predicted_route = crisp.generate_route_for_inference(vars_predict.detach().numpy(), visit_z0)
        else:
            predicted_route = data_utils.score_to_routes(vars_predict.detach().numpy().squeeze(),
                                                         real_paths.transpose()[0],
                                                         use_post_process_type2)
            if use_post_process_type1:
                visited = set()
                dup_removd_route = []
                for x in predicted_route:
                    if x not in visited:
                        dup_removd_route.append(x)
                    else:
                        dup_removd_route.append(0)
                predicted_route = dup_removd_route

        real_route = real_paths.transpose()
        is_valid = prob.valid_routes([predicted_route], visit)[0]
        num_valid_routes.append(is_valid)
        if is_valid:
            norm_reward.append(prob.norm_reward([predicted_route], real_route))
        real_routes.append(list(real_route.squeeze()))
        predicted_routes.append(predicted_route)

    print('real={}'.format(real_routes[:10]))
    print('gen={}'.format(predicted_routes[:10]))
    print('valid routes={}'.format(sum(num_valid_routes) /(len(num_valid_routes)+0.0001)))
    print('norm-reward={}'.format(sum(norm_reward) /(len(norm_reward)+0.0001)))



def train(data_file, prob, train_batch_size, max_width, testing_set_size, d_n_hidden, g_n_hidden, g_n_input,
          lr, g_lr, use_crisp, use_post_process_type1, use_post_process_type2):
    # prepare the CRISP module
    data_gen = data_utils.ScheduleDataGen(data_file, prob.max_stops, prob.num_locs)
    crisp = CRISP(prob.num_locs, prob.max_stops, max_width)
    # exit()

    # generators
    gnt = generator(hidden_dim=g_n_hidden, z_dim=g_n_input + prob.num_locs, val_dim=prob.num_locs,
                    n_vars=prob.max_stops + 1)

    # discriminators
    dmt = discriminator(hidden_dim=d_n_hidden, val_dim=prob.num_locs)

    d_label_1 = torch.ones(train_batch_size)
    d_label_0 = torch.zeros(train_batch_size)
    d_label = torch.cat((d_label_1, d_label_0), 0)

    d_optim = torch.optim.Adam(dmt.parameters(), lr=lr)
    g_optim = torch.optim.Adam(gnt.parameters(), lr=g_lr)

    for epoch in range(1, 50):
        print("[ {} iterations]".format(epoch))
        for it in range(100):
            vars_real, visit, real_paths = data_gen.next_data(train_batch_size)
            vars_real = torch.Tensor(vars_real).float()
            visit_z0 = torch.Tensor(visit).float()

            visit_z = visit_z0.repeat(prob.max_stops + 1, 1, 1)
            # the random variable z
            z = torch.randn([prob.max_stops + 1, train_batch_size, g_n_input])
            z = torch.cat((visit_z, z), 2)

            out = gnt(z)
            vars_predict = F.softmax(out, dim=2)
            if use_crisp:
                out_mask = crisp.generate_mask_with_ground_truth(real_paths, visit)
                out_mask = torch.from_numpy(out_mask)
                # mask
                # print("out mask:",out_mask)
                vars_predict = torch.mul(vars_predict, out_mask)
                # print("after masking:",vars_predict)
                # renormalize
                norm_vars_predict = vars_predict/torch.sum(vars_predict, dim=2, keepdim=True)
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
        test(data_gen, testing_set_size, g_n_input, crisp, use_crisp,
             use_post_process_type1, use_post_process_type2, gnt)


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
                        help="whether to use crisp or not (default=False: no crisp).")
    parser.add_argument("--use_post_process_type1", type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use post processing for baseline (default=False: no post processing).")
    parser.add_argument("--use_post_process_type2", type=str2bool, nargs='?', const=True, default=False,
                        help="whether to use post processing for baseline (default=False: no post processing).")

    args = parser.parse_args()
    print("read problem")
    prob = pickle.load(open(args.prob_file, 'rb'))
    train(args.data_file, prob, args.batchnum, args.max_width, args.batchnum_summary, args.d_n_hidden, args.g_n_hidden,
          args.g_n_input, args.lr, args.glr, args.use_crisp, args.use_post_process_type1, args.use_post_process_type2)
