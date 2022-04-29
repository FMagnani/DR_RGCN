from dgl.data.knowledge_graph import KnowledgeGraphDataset

class DRKGDataset(KnowledgeGraphDataset):
    
    def __init__(self, reverse=True, raw_dir=None, force_reload=False,
                 verbose=True, transform=None):
        name="DRKG"
        super(DRKGDataset, self).__init__(name, reverse, raw_dir,
                                          force_reload, verbose, transform)

    def __getitem__(self, idx):
        return super(DRKGDataset, self).__getitem__(idx)
    
    def __len__(self):
        return super(DRKGDataset, self).__len__()

    def download(self):
        pass

