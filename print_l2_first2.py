import features
from serialize import NNUEReader, NNUEWriter


def print_spsa_params(nnue_filename):
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        model = NNUEReader(f, feature_set).model

    c_end = 6
    c_end_biases = 128

    num_weights = num_biases = 0
    stack_range = range(2)

    # L2 weights - 8 x 960 = 7680
    for i in stack_range:
        for j in range(32):
            for k in range(30):
                value = int(model.layer_stacks.l2.weight[32 * i + j, k] * 64)
                if abs(value) >= 30:
                    print(f"twoW[{i}][{j}][{k}],{value},-127,127,{c_end},0.0020")
                    num_weights += 1

    # L2 biases - 8 x 32 = 256
    # for i in stack_range:
    #     for j in range(32):
    #         value = int(model.layer_stacks.l2.bias[32 * i + j] * 64 * 127)
    #         print(f"twoB[{i}][{j}],{value},-16384,16384,{c_end_biases},0.0020")
    #         num_biases += 1

    print(f"# weights: {num_weights}")
    print(f"# biases:  {num_biases}")
    print(f"total:     {num_weights + num_biases}")


if __name__ == "__main__":
    # filename = "nnue/nn-ddcfb9224cdb.nnue"
    # filename = "nnue/nn-74f1d263ae9a.nnue"
    filename = "nnue/nn-e8bac1c07a5a.nnue"
    print_spsa_params(filename)
