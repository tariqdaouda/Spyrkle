import notebook
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
            """
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


def get_children(module) :
    return module.children()

def get_name(module) :
    return module.__class__.__name__

dag = notebook.DagreDAG(BOOK, "Network")
dag.derive(model, get_children, get_name)

print(dag.nodes)
print(dag.edges)

# print(model.modules())

BOOK.save()