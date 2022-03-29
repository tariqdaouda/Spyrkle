from .graph import Abstract_GraphCrawler
import hashlib

class pyTorchCrawler(Abstract_GraphCrawler):
    """
    Custom grpah crawler from building graphs out of pyTorch models.
    
    model: The model you want to visualize
    inputs: Example inputs to the model. Should of the right shape.
    optimizer: If a pyToch optimizer, will add optimizer information as graph attribute
    ignore_nodes: List of node names that should be ignored, ex: relu, Linear, ...
    strict_ignore: if true will look for exact matches
    ignore_empty_scopes: Ignore all the nodes with empty scopes. Simplifies the graph a lot, turn it off for special debugging purposes.
    onnx_translations: Replace the ONNX names for some layers by custom ones
    """
    
    def __init__(self, model, inputs, optimizer=None, ignore_nodes=None, strict_ignore=False, ignore_empty_scopes=False, ignore_params=True, onnx_translations={"Gemm": "Fully connected"}):
        import torch
        
        self.trace, out = torch.jit._get_trace_graph(model, inputs)
        # torch.onnx._optimize_trace(self.trace, torch.onnx.OperatorExportTypes.ONNX)
        # print(self.trace)
        # self.torch_graph = self.trace.graph()
        self.torch_graph = self.trace
        # self.all_nodes = self.torch_graph.nodes()
        self.all_nodes = list( self.torch_graph.nodes() )
        # print("===", self.all_nodes[0].pyname)

        super(pyTorchCrawler, self).__init__(roots=self.all_nodes, parents_to_children=False)
        self.model = model
        self.optimizer = optimizer

        if not ignore_nodes :
            self.ignore_nodes = []
        else :
            self.ignore_nodes = ignore_nodes

        self.ignore_empty_scopes = ignore_empty_scopes
        self.ignore_params = ignore_params
        self.onnx_translations = onnx_translations
        self.strict_ignore = strict_ignore

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
        """return the next nodes"""
        return [o.node() for o in node.inputs()]

    def _should_ignore(self, kind) :
        """returns true if the node should be ignored"""
        if self.strict_ignore:
            last = kind.split("::")[-1]
            return last in self.ignore_nodes
        else :
            for typ in self.ignore_nodes:
                if len(typ) < 2:
                    raise ValueError("ignore node len < 2, this not allowed (val: %s)." % typ)
                if kind.find(typ) > -1:
                    return True

        return False

    def get_node_uid(self, node) :
        """return a nodes unique id"""
        kind = node.kind()
        
        if (self.ignore_empty_scopes and len(kind) == 0) or (self.ignore_params and self.get_node_type(node).lower() == "param") or (self._should_ignore(kind) ) :
            return None
        # return kind + ">(" + "-".join([o.uniqueName() for o in node.outputs()]) + ")"
        uid =kind + ">(" + "-".join([ hashlib.md5(str(o).encode("utf-8")).hexdigest() for o in node.outputs()]) + ")"
        return uid

    def get_node_type(self, node) :
        """return the type of the node"""
        base_name = node.kind()[6:]
        try :
            base_name = self.onnx_translations[base_name]
        except KeyError :
            pass

        if base_name.lower() == "pythonop":
            base_name = node.pyname()
        
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