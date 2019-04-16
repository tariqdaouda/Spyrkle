import Spyrkle.notebook as notebook
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BOOK = notebook.Notebook("Test pyTorch Notebook")

# Define model
class TheModelClass(nn.Module):
    def __init__(self):
        global BOOK
        notes = notebook.Notes(BOOK, "Model")
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

#These are the function needed to make the graph

def get_graph(model, inputs) :
    trace, out = torch.jit.get_trace_graph(model, inputs)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    torch_graph = trace.graph()
    return torch_graph

#node label
def get_name(node) :
    base_name = node.scopeName()[len(model.__class__.__name__)+1:]
    if len(base_name) == 0 :
        base_name = node.kind()[len("onnx::"):]

    return base_name

#node uniqye id
def get_uid(node) :
    #ignore constant and paran nodes
    if get_name(node).lower() in ["constant", "param"] :
        return None
    return node.scopeName() + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"

#find the nest nodes
def get_next(node):
    return [o.node() for o in node.inputs()]

#get the attributes of each node
def get_node_attributes(node) :
    return {k: node[k] for k in node.attributeNames()}

#set some parameters ex: for style
def get_node_params(node) :
    if get_name(node).lower().find("relu") > -1 :
        return {"class": "relu"}
    return {}

if __name__ == '__main__':
    # Initialize model
    model = TheModelClass()
    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #pytroch needs inputs to derive the graph
    inputs = torch.FloatTensor( numpy.random.random((6, 3, 32, 32)) )

    torch_graph = get_graph(model, inputs)
    #add a DAG page
    dag = notebook.DagreDAG(BOOK, "Network")
    
    #craql the DAG and find the structure
    dag.crawl(
        torch_graph.nodes(),
        get_next,
        get_uid,
        get_name,
        get_node_parameters_fct=get_node_params,
        get_node_attributes_fct = get_node_attributes,
        parents_to_children=False,
        autoincrement_names=False
    )

    #adding somecolors
    dag.set_css_rule(".relu", ("fill: #00ffd0", ) )

    #ass the infos of the optimiser to the graph
    opt_attr = optimizer.state_dict()["param_groups"][0]
    del opt_attr['params']
    dag.set_attributes(opt_attr)

    #lets' add a caption
    dag.set_caption("This is a model taken from a pyTorch tutorial. It's a basic conv net")

    #FINISHING WITH SOM NONSENSE
    notes = notebook.Notes(BOOK, "Notes on life")
    for i in range(10) :
        notes.add_note("Note %s" % i, "Life is %s Lorem ipsum dolor sit amet, consectetur adipisicing elit. Alias dolorum asperiores at veritatis architecto sequi nulla perspiciatis rerum modi, repellat assumenda quisquam dolorem sit molestiae aspernatur cum nemo placeat laboriosam." % i)

    notes = notebook.Notes(BOOK, "Notes on life 2")
    for i in range(10) :
        notes.add_note("Note %s" % (i+10), "Life is %s Lorem ipsum dolor sit amet, consectetur adipisicing elit. Alias dolorum asperiores at veritatis architecto sequi nulla perspiciatis rerum modi, repellat assumenda quisquam dolorem sit molestiae aspernatur cum nemo placeat laboriosam." % i)

    notes = notebook.Notes(BOOK, "Notes on life 3")
    for i in range(10) :
        notes.add_bullet_points_note("Note %s" % (i+100), ["test", "text", "iop"])


BOOK.save(overwrite=True)
