import Spyrkle.notebook as notebook
import Spyrkle.pages.core_pages as pages
import Spyrkle.pages.graph_pages as graphs
import Spyrkle.pages.graphs_more as gmore

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define model
class TheModelClass(nn.Module):
    def __init__(self):
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

    BOOK = notebook.Notebook("Test pyTorch Notebook Lean")
    
    #add a DAG page
    dag = graphs.DagreGraph(BOOK, "ConvNet")
    
    #crawl the DAG and find the structure
    dag.crawl(
        CustomCrawler(model, inputs, optimizer=optimizer),
        autoincrement_names=False,
    )

    #adding somecolors
    dag.set_css_rule(".relu", ("fill: #00ffd0", ) )

    #lets' add a caption
    dag.set_caption("This is a model taken from a pyTorch tutorial. It's a basic conv net.")

    BOOK.save(overwrite=True)
