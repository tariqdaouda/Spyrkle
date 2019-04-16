import notebook
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BOOK = notebook.Notebook("test_pytorch_notebook")

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

# Initialize model
model = TheModelClass()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def get_name(node) :
    base_name = node.scopeName()[len(model.__class__.__name__)+1:]
    if len(base_name) == 0 :
        base_name = node.kind()[len("onnx::"):]

    return base_name

def get_uid(node) :
    if get_name(node).lower() in ["constant", "param"] :
        return None
    return node.scopeName() + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"

def get_next(node):
    return [o.node() for o in node.inputs()]

def get_node_attributes(node) :
    return {k: node[k] for k in node.attributeNames()}

def get_node_params(node) :
    if get_name(node).lower().find("relu") > -1 :
        return {"class": "conv"}
    return {}

inputs = torch.FloatTensor( numpy.random.random((6, 3, 32, 32)) )

trace, out = torch.jit.get_trace_graph(model, inputs)
torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
torch_graph = trace.graph()

dag = notebook.DagreDAG(BOOK, "Network")
dag.crawl(torch_graph.nodes(), get_next, get_uid, get_name, get_node_parameters_fct=get_node_params, get_node_attributes_fct = get_node_attributes, parents_to_children=False, autoincrement_names=False)
dag.set_css_rule(".conv", ("fill: #00ffd0", ) )

opt_attr = optimizer.state_dict()["param_groups"][0]
del opt_attr['params']
dag.set_attributes(opt_attr)

# print(dag.nodes)
# print("-------------")
# print(dag.edges)



BOOK.save(overwrite=True)