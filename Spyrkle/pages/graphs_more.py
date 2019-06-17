from .graph_pages import GraphCrawler

class pyTorchCrawler(GraphCrawler):
    """docstring for pyTorchCrawler"""
    
    def __init__(self, model, inputs, ignore_nodes=None, remove_nameless_scopes=True, onnx_translations={"Gemm": "Fully connected"}):
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

        self.remove_nameless_scopes = remove_nameless_scopes
        self.onnx_translations = onnx_translations

    def get_next(self, node) :
        return [o.node() for o in node.inputs()]

    def _should_ignore(self, scopeName) :
        last = scopeName.split("/")[-1]
        return last in self.ignore_nodes

    def get_node_uid(self, node) :
        scopeName = node.scopeName()
        if (self.remove_nameless_scopes and len(scopeName) == 0) or (self._should_ignore(scopeName) ) :
            return None
        return scopeName + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"
    
    def get_node_label(self, node) :
        base_name = node.kind()[6:]
        try :
            base_name = self.onnx_translations[base_name]
        except KeyError :
            pass
        return base_name
        
    def get_node_shape(self, node):
        import re
        #hacky but works for now
        a = re.search(": (.+) =", str(node))
        return a.group(1)

    def get_node_parameters(self, node):
        return {}

    def get_edge_parameters(self, e0, e1):
        return {}
    
    def get_node_attributes(self, node):
        ret = {k: node[k] for k in node.attributeNames()}
        ret["path"] = "<p>" + node.scopeName().replace("/", "<br>") + "</p>"
        ret["shape"] = self.get_node_shape(node)
        return ret
    
    def get_edge_attributes(self, e0, e1):
        return {}