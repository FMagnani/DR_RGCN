"""
Differences compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR.
"""

import argparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from load_drkg import DRKGDataset
from dgl.dataloading import GraphDataLoader

from link_utils import preprocess, SubgraphIterator, calc_mrr, compute_hits, compute_results
from model import RGCN

class LinkPredict(nn.Module):
                                  # def: h dim 500, num bases 100
    def __init__(self, in_dim, num_rels, h_dim=400, num_bases=200, dropout=0.2, reg_param=0.01):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, regularizer="bdd",
                         num_bases=num_bases, dropout=dropout, self_loop=True)
        self.dropout = nn.Dropout(dropout)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(th.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = th.sum(s * r * o, dim=1)
        return score

    def forward(self, g, nids):
        return self.dropout(self.rgcn(g, nids=nids))

    def regularization_loss(self, embedding):
        return th.mean(embedding.pow(2)) + th.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def main(args):

    # Set the absolute path to the dataset folder. The dataset should be formatted into the following files:
    # train.txt, test.txt, valid.txt, entities.dict, relations.dict (all these are tsv)
    raw_dir = "/mnt/raid1/fede/"

    data = DRKGDataset(reverse=False, raw_dir=raw_dir)
    graph = data[0]
    num_nodes = graph.num_nodes()
    num_rels = data.num_rels

    train_g, test_g, valid_g = preprocess(graph, num_rels)
    test_nids = th.arange(0, num_nodes)
    test_mask = graph.edata['test_mask']

    valid_nids = th.arange(0, valid_g.num_nodes())

    # Set here the batch size (for edge sampling) as sample_size, and the epochs for training
                                                            # Default: sample_size 30000, num_epochs 6000
    subg_iter = SubgraphIterator(train_g, num_rels, args.edge_sampler, sample_size=15000, num_epochs=500)
    dataloader = GraphDataLoader(subg_iter, batch_size=1, collate_fn=lambda x: x[0])

    # Prepare data for metric computation
    src, dst = graph.edges()
    triplets = th.stack([src, graph.edata['etype'], dst], dim=1)

    model = LinkPredict(num_nodes, num_rels)
    # Set LR                              default lr 1e-2
    optimizer = th.optim.Adam(model.parameters(), lr=1e-2)

    if args.gpu >= 0 and th.cuda.is_available():
        device = th.device(args.gpu)
    else:
        device = th.device('cpu')
    model = model.to(device)

    model_state_file = 'model_state.pth'
    best_mrr = 0
    
    resume_training = False
    if resume_training:
        th.load('model_state.pth')
    
    for epoch, batch_data in enumerate(dataloader):
        model.train()

        g, train_nids, edges, labels = batch_data
        g = g.to(device)
        train_nids = train_nids.to(device)
        edges = edges.to(device)
        labels = labels.to(device)

        embed = model(g, train_nids)
        loss = model.get_loss(embed, edges, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clip gradients
        optimizer.step()

        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f}".format(epoch, loss.item(), best_mrr))


    print("Saving... ("+str(epoch)+" epochs)")
    th.save({'state_dict': model.state_dict(), 'epoch': epoch}, 'model_state.pth')

    # Compute hits - on cpu
    model = model.cpu()
    model.eval()
    embed = model(valid_g, valid_nids)
    
    sources, targets = valid_g.edges()
    rels = valid_g.edata["_TYPE"]
    
    triplets = th.stack([sources,rels,targets], dim=1)
    
    with th.no_grad():
        scores = model.calc_score(embed, triplets)
    
    scores = scores.numpy()

    results = compute_results(raw_dir, scores)

    print(results)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for link prediction')
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--eval-protocol", type=str, default='filtered',
                        choices=['filtered', 'raw'],
                        help="Whether to use 'filtered' or 'raw' MRR for evaluation")
    parser.add_argument("--edge-sampler", type=str, default='neighbor',
                        choices=['uniform', 'neighbor'],
                        help="Type of edge sampler: 'uniform' or 'neighbor'"
                             "The original implementation uses neighbor sampler.")

    args = parser.parse_args()
    print(args)
    main(args)
