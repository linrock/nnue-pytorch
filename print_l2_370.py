import features
from serialize import NNUEReader, NNUEWriter


def print_spsa_params(nnue_filename):
    feature_set = features.get_feature_set_from_name("HalfKAv2_hm")
    with open(nnue_filename, "rb") as f:
        model = NNUEReader(f, feature_set).model

    c_end = 6
    for i in range(8):
        for j in range(32):
            for k in range(30):
                value = int(model.layer_stacks.l2.weight[32 * i + j, k] * 64)
                if abs(value) >= 50:
                    print(f"twoW[{i}][{j}][{k}],{value},-127,127,{c_end},0.0020")


if __name__ == "__main__":
    filename = "nnue/nn-ddcfb9224cdb.nnue"
    filename = "nnue/nn-74f1d263ae9a.nnue"
    # filename = "nnue/nn-e8bac1c07a5a.nnue"
    print_spsa_params(filename)
