from .graph_pages import GraphCrawler

class pyTorchCrawler(GraphCrawler):
    """docstring for pyTorchCrawler"""
    
    def __init__(self, model, inputs, ignore_nodes=None):
        import torch
        self.trace, out = torch.jit.get_trace_graph(model, inputs)
        torch.onnx._optimize_trace(self.trace, torch.onnx.OperatorExportTypes.ONNX)
        self.torch_graph = self.trace.graph()
        self.all_nodes = self.torch_graph.nodes()

        self.all_nodes = list( self.torch_graph.nodes() )
        super(pyTorchCrawler, self).__init__(roots=self.all_nodes, parents_to_children=False)
        self.model = model
        
        if not ignore_nodes :
            self.ignore_nodes = []
        else :
            self.ignore_nodes = ignore_nodes

    def get_next(self, node) :
        return [o.node() for o in node.inputs()]

    def get_node_uid(self, node) :
        # if len(node.scopeName()) == 0 or self.ignore_nodes :
        if self.ignore_nodes :
            return None
        return node.scopeName() + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"
    
    def get_node_label(self, node) :
        base_name = node.scopeName()[len(self.model.__class__.__name__)+1:]
        if len(base_name) == 0 :
            base_name = node.kind()[len("onnx::"):]
        return base_name
    
    def get_node_parameters(self, node):
        return {}

    def get_edge_parameters(self, e0, e1):
        return {}
    
    def get_node_attributes(self, node):
        return {k: node[k] for k in node.attributeNames()}
    
    def get_edge_attributes(self, e0, e1):
        return {}
