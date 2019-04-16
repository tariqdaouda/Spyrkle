import Spyrkle.notebook as notebook
import Spyrkle.pages.core_pages as pages
import Spyrkle.pages.graph_pages as graphs
import Spyrkle.pages.graphs_more as gmore

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BOOK = notebook.Notebook("Test pyTorch Notebook2")

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        global BOOK
        notes = pages.Notes(BOOK, "Model")
        notes.add_note("Description", "This model comes from this page: https://pytorch.org/tutorials/beginner/saving_loading_models.html")
        notes.add_note(
            "Code",
            "These are the first layers",
            code = """
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
            """.replace("                ", "")
        )

        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomCrawler(gmore.pyTorchCrawler):
    """docstring for CustomCrawler"""
        
    #set some parameters ex: for style
    def get_node_parameters(self, node) :
        if self.get_node_label(node).lower().find("relu") > -1 :
            return {"class": "relu"}
        return {}

if __name__ == '__main__':
    # Initialize model
    model = TheModelClass()
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #pytroch needs inputs to derive the graph
    inputs = torch.FloatTensor( numpy.random.random((6, 3, 32, 32)) )

    # torch_graph = get_graph(model, inputs)
    #add a DAG page
    dag = graphs.DagreGraph(BOOK, "Network")
    
    #crawl the DAG and find the structure
    dag.crawl(
        CustomCrawler(model, inputs),
        autoincrement_names=False
    )

    #adding somecolors
    dag.set_css_rule(".relu", ("fill: #00ffd0", ) )

    #ass the infos of the optimiser to the graph
    opt_attr = { "SGD-%s" % k : v for k, v in optimizer.state_dict()["param_groups"][0].items() }
    del opt_attr['SGD-params']
    dag.set_attributes(opt_attr)

    #lets' add a caption
    dag.set_caption("This is a model taken from a pyTorch tutorial. It's a basic conv net")

    #FINISHING WITH SOME NONSENSE
    notes = pages.Notes(BOOK, "Notes on life")
    for i in range(10) :
        notes.add_note("Note %s" % i, "Life is %s Lorem ipsum dolor sit amet, consectetur adipisicing elit. Alias dolorum asperiores at veritatis architecto sequi nulla perspiciatis rerum modi, repellat assumenda quisquam dolorem sit molestiae aspernatur cum nemo placeat laboriosam." % i)

    notes = pages.Notes(BOOK, "Notes on life 2")
    for i in range(10) :
        notes.add_note("Note %s" % (i+10), "Life is %s Lorem ipsum dolor sit amet, consectetur adipisicing elit. Alias dolorum asperiores at veritatis architecto sequi nulla perspiciatis rerum modi, repellat assumenda quisquam dolorem sit molestiae aspernatur cum nemo placeat laboriosam." % i)

    notes = pages.Notes(BOOK, "Notes on life 3")
    for i in range(10) :
        notes.add_bullet_points_note("Note %s" % (i+100), ["test", "text", "iop"])


BOOK.save(overwrite=True)
