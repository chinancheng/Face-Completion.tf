import matplotlib
matplotlib.use('Agg')
from train import Train
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', help='path of the data',
                        default='./data/', type=str)
    parser.add_argument('-m', '--model_path', dest='model_path', help='path of the model folder',
                        default='./model/', type=str)
    parser.add_argument('-o', '--output_path', dest='output_path', help='path of the output folder',
                        default='./out', type=str)
    parser.add_argument('-g', '--graph_path', dest='graph_path', help='path of the graph',
                        default='./graph', type=str)
    parser.add_argument('-b', '--batch_size', dest='batch_size', help='batch size',
                        default=32, type=int)
    parser.add_argument('-e', '--epoch', dest='epoch', help='epoch',
                        default=20, type=int)
    parser.add_argument('-r', '--restore', dest='restore', help='restore',
                        default=False, type=bool)
    args = parser.parse_args()

    return args

args = parse_args() 

if __name__ == '__main__':
    print(args)
    epoch = args.epoch
    batch_size = args.batch_size
    data_path = args.data_path
    model_path = args.model_path
    output_path = args.output_path
    graph_path = args.graph_path
    restore = args.restore
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    train = Train(epoch, batch_size, data_path, model_path, output_path, graph_path, restore)
    train.train()

                            
