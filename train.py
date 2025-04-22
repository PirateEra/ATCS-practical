
import argparse
from models import SNLIModel

def parse_args():
    parser = argparse.ArgumentParser(description="NLI training task")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--encoder_model", type=str, default="uni-lstm", help="Choices would be, uni-lstm, bi-lstm, bi-max-lstm, mean")
    parser.add_argument("--epochs", type=int, default=20) # It converges based on the LR so best to let it run for a while with a higher value
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout_encoder", type=float, default=0.0)
    parser.add_argument("--dropout_classifier", type=float, default=0.0)
    parser.add_argument("--lstm_dim", type=int, default=512)
    parser.add_argument("--classifier_dim", type=int, default=512)
    parser.add_argument("--class_count", type=int, default=3)
    parser.add_argument("--sgd_lr", type=float, default=0.1)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=0.99)
    # Paths
    parser.add_argument("--outputdir", type=str, default='modelsaves/')
    parser.add_argument("--checkpoint_path", type=str, default='none')
    parser.add_argument("--outputmodelname", type=str, default='model_output')
    parser.add_argument("--glovepath", type=str, default='glove.840B.300d.txt')
    # This one is for evaluation
    parser.add_argument("--sentevalpath", type=str, default='../SentEval')

    args, _ = parser.parse_known_args()

    return args

if __name__ == "__main__":
    params = parse_args()
    model = SNLIModel(params)
    model.train()
