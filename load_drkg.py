from dgl.data.knowledge_graph import KnowledgeGraphDataset

"""
    Class for using a personal dataset. 
    The dataset should be stored into a folder with name "DRKG", that is in turn into "raw_dir".
    The dataset needs to be composed of the following files:
        - train.txt, test.txt, valid.txt (that in fact are tsv)
        - entities.dict, relations.dict (that also are tsv)
"""    

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
#    We are loading the graph through the code originally used for the built-in datasets, that are donwloaded.
#    Since we already have our dataset in the folder, we disable the download step.
        pass

