from .graph_pages import Abstract_GraphCrawler

class pyTorchCrawler(Abstract_GraphCrawler):
    """
    Custom grpah crawler from building graphs out of pyTorch models.
    
    model: The model you want to visualize
    inputs: Example inputs to the model. Should of the right shape.
    optimizer: If a pyToch optimizer, will add optimizer information as graph attribute
    ignore_nodes: List of node names that should be ignored, ex: relu, Linear, ...
    ignore_empty_scopes: Ignore all the nodes with empty scopes. Simplifies the graph a lot, turn it off for special debugging purposes.
    onnx_translations: Replace the ONNX names for some layers by custom ones
    """
    
    def __init__(self, model, inputs, optimizer=None, ignore_nodes=None, ignore_empty_scopes=True, onnx_translations={"Gemm": "Fully connected"}):
        import torch
        self.trace, out = torch.jit.get_trace_graph(model, inputs)
        torch.onnx._optimize_trace(self.trace, torch.onnx.OperatorExportTypes.ONNX)
        self.torch_graph = self.trace.graph()
        self.all_nodes = self.torch_graph.nodes()

        self.all_nodes = list( self.torch_graph.nodes() )
        
        super(pyTorchCrawler, self).__init__(roots=self.all_nodes, parents_to_children=False)
        self.model = model
        self.optimizer = optimizer

        if not ignore_nodes :
            self.ignore_nodes = []
        else :
            self.ignore_nodes = ignore_nodes

        self.ignore_empty_scopes = ignore_empty_scopes
        self.onnx_translations = onnx_translations

    def get_graph_prameter_size(self):
        """Return the total number of parameters in the graph"""
        total_sum = 0
        for p in self.model.parameters():
            param_total = 1
            for size in p.size():
                param_total *= size
            total_sum += param_total
        return total_sum

    def get_optimizer_info(self):
        opt_attr = { "%s-%s" % (self.optimizer.__class__.__name__, k) : v for k, v in self.optimizer.state_dict()["param_groups"][0].items() }
        del opt_attr['%s-params' % self.optimizer.__class__.__name__]
        return opt_attr

    def get_graph_attributes(self):
        """Return a dict representing the attributes of the graph"""
        res = {
            "# parameters": self.get_graph_prameter_size()
        }
        if self.optimizer :
            res.update(self.get_optimizer_info())
        return res
    
    def get_next(self, node) :
        """return the next node"""
        return [o.node() for o in node.inputs()]

    def _should_ignore(self, scopeName) :
        """returns true if the node should be ignored"""
        last = scopeName.split("/")[-1]
        return last in self.ignore_nodes

    def get_node_uid(self, node) :
        """return a nodes unique id"""
        scopeName = node.scopeName()
        if (self.ignore_empty_scopes and len(scopeName) == 0) or (self._should_ignore(scopeName) ) :
            return None
        return scopeName + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"
    
    def get_node_type(self, node) :
        """return the type of the node"""
        base_name = node.kind()[6:]
        try :
            base_name = self.onnx_translations[base_name]
        except KeyError :
            pass
        return base_name

    def get_node_label(self, node):
        """return node label"""
        return self.get_node_type(node) + " " + self.get_node_simplified_shape(node)

    def get_node_simplified_shape(self, node):
        """return a simplified version of the node shape"""
        shape = self.get_node_shape(node)
        if shape.find("Float") == 0 :
            return shape[5:]
        else :
            return shape

    def get_node_shape(self, node):
        """return the node shape"""
        import re
        #hacky but works for now
        a = re.search(": (.+) =", str(node))
        return a.group(1)

    def get_node_parameters(self, node):
        """return node svg, css, dagre-d3 parameters"""
        return {}

    def get_edge_parameters(self, e0, e1):
        """return edge svg, css, dagre-d3 parameters"""
        return {}
    
    def get_node_attributes(self, node):
        """return node custom attributes"""
        ret = {k: node[k] for k in node.attributeNames()}
        ret["path"] = "<p>" + node.scopeName().replace("/", "<br>") + "</p>"
        ret["shape"] = self.get_node_shape(node)
        return ret
    
    def get_edge_attributes(self, e0, e1):
        """return edge custom attributes"""
        return {}