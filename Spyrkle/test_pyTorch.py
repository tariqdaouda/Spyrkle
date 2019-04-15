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
        # global BOOK
        # notes = notebook.Notes(BOOK, "Model")
        # notes.add_note("Description", "This model comes from this page: https://pytorch.org/tutorials/beginner/saving_loading_models.html")
        # notes.add_note(
        #     "Code",
        #     "These are the first layers",
        #     code = """
        #         self.conv1 = nn.Conv2d(3, 6, 5)
        #         self.pool = nn.MaxPool2d(2, 2)
        #         self.conv2 = nn.Conv2d(6, 16, 5)
        #     """.replace("                ", "")
        # )

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

############################


def pyTorchParse_bck(model, *inputs, **kw_inputs) :
    trace, out = torch.jit.get_trace_graph(model, *inputs, **kw_inputs)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    torch_graph = trace.graph()

    print(torch_graph)
    print(out.shape)
    
    for node in torch_graph.nodes():
        print("------------")
        print(node.scopeName())
        print(node.kind())
        params = {k: node[k] for k in node.attributeNames()} 
        print(params)
        outputs = [o.unique() for o in node.outputs()]
        print(outputs)
        inputs = [o.unique() for o in node.inputs()]
        print(inputs)
        print(dir(node))
        print(node.outputsSize())
        print(node.output())
        
    print("------------")
    inputs = [o.unique() for o in torch_graph.inputs()]
    print(inputs)

# def pyTorchParse(model, *inputs, **kw_inputs) :
#     def _get_name(node, nodes) :
#         base_name = node.scopeName()[len(model.__class__.__name__)+1:]
#         if len(base_name) == 0 :
#             base_name = node.kind()[len("onnx::"):]

#         return notebook.Abstract_DAG.resolve_node_name( base_name, nodes, False )

#     trace, out = torch.jit.get_trace_graph(model, *inputs, **kw_inputs)
#     torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
#     torch_graph = trace.graph()

#     nodes = {}
#     edges = set()
    
#     for node in torch_graph.nodes():
#         node_name = _get_name(node, nodes)
#         node_id = node_name + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"
#         print(id(node), node_id, node_name)
#         nodes[id(node)] = {"label": node_name}
    
#     print(nodes)

def get_uid(node) :
    return node.scopeName() + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"

def get_name(node) :
    base_name = node.scopeName()[len(model.__class__.__name__)+1:]
    if len(base_name) == 0 :
        base_name = node.kind()[len("onnx::"):]

    return base_name

def get_next(node):
    # print(node, [o.node() for o in node.outputs()])
    return [o.node() for o in node.inputs()]

notes0 = notebook.Notes(BOOK, "Model2")

inputs = torch.FloatTensor( numpy.random.random((6, 3, 32, 32)) )
# pyTorchParse(model, inputs)

# print(model.forward(inputs))

trace, out = torch.jit.get_trace_graph(model, inputs)
torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
torch_graph = trace.graph()

dag = notebook.DagreDAG(BOOK, "Network")
dag.crawl(torch_graph.nodes(), get_next, get_uid, get_name, parents_to_children=False, autoincrement_names=False)

print(dag.nodes)
print("-------------")
print(dag.edges)


# notes = notebook.Notes(BOOK, "Model")
# notes.add_note("Description", "This model comes from this page: https://pytorch.org/tutorials/beginner/saving_loading_models.html")
# notes.add_note(
#     "Code",
#     "These are the first layers",
#     code = """
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#     """
# )

BOOK.save(overwrite=True)